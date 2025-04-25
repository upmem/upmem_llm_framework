"""Implements all profiling related classes and functions."""

import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch.nn import Conv2d, Dropout, Embedding, LayerNorm, Linear, SiLU, Softmax
from transformers.activations import NewGELUActivation
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.pytorch_utils import Conv1D

from upmem_llm_framework.options import Options
from upmem_llm_framework.simulator import Simulator
from upmem_llm_framework.utils import LayerProfile, add_dictionaries


@dataclass
class LayerLog:
    """Class to store profiling information for a layer."""

    id: int
    name: str
    context: str
    summarization: bool
    start_time: float
    input: torch.Size
    weights: torch.Size
    output: torch.Size
    exec_time_ms: float
    performance: dict[str, float]
    energy: dict[str, float]
    transfer_bytes: dict[str, float]


@dataclass
class UPMProfiler:
    """Class to profile the execution of a model."""

    options: Options
    n_layers: int = 0
    n_executions: int = 0
    layers: dict[torch.nn.Module | Callable, LayerProfile] = field(default_factory=dict)
    functions: dict[str, LayerProfile] = field(default_factory=dict)

    sim_compute: bool = False
    sim_sliding_window: int | None = None
    sim_num_key_value_heads: int | None = None
    sim_data_type: str = "float16"
    sim_data_type_bytes: float = 2.0
    simulator: Simulator | None = None

    start_inference: float = field(default_factory=time.time_ns)
    inference_time: float = 0
    summarization_time: float = 0
    sum_perf: dict[str, float] = field(default_factory=dict)
    gen_perf: dict[str, float] = field(default_factory=dict)
    sum_energy: dict[str, float] = field(default_factory=dict)
    gen_energy: dict[str, float] = field(default_factory=dict)
    sum_transfer_bytes: dict[str, float] = field(default_factory=dict)
    gen_transfer_bytes: dict[str, float] = field(default_factory=dict)

    last_layer: str = "lm_head"
    batch_size: int = 1

    layers_start: dict[torch.nn.Module, float] = field(default_factory=dict)
    layers_end: dict[torch.nn.Module, float] = field(default_factory=dict)
    log: list[LayerLog] = field(default_factory=list)

    forward_input_shape: torch.Size = field(default_factory=torch.Size)
    forward_time_start: float = 0
    forward_time_end: float = 0

    func_input_shape: torch.Size = field(default_factory=torch.Size)
    start_func: float = 0
    end_func: float = 0

    @staticmethod
    def _fixed_layer_dim(layer: torch.nn.Module) -> tuple[int, int]:
        """Return a constant size for the layer."""
        del layer
        return (1, 1)

    layer_dimensions: ClassVar[Mapping[type, Callable[[torch.nn.Module], tuple[int, int]]]] = {
        Linear: (lambda layer: (layer.in_features, layer.out_features)),
        NewGELUActivation: _fixed_layer_dim,
        SiLU: _fixed_layer_dim,
        LlamaRMSNorm: (lambda layer: (layer.weight.size()[0], layer.weight.size()[0])),
        LlamaRotaryEmbedding: _fixed_layer_dim,
        LayerNorm: _fixed_layer_dim,
        Embedding: _fixed_layer_dim,
        Dropout: _fixed_layer_dim,
        Softmax: (lambda layer: (layer.dim, layer.dim)),
        Conv1D: (lambda layer: (layer.weight.shape[0], layer.weight.shape[1])),
        Conv2d: (lambda layer: (layer.kernel_size[0], layer.kernel_size[1])),
    }

    functional_layers = (LlamaRMSNorm, SiLU)

    def __post_init__(self) -> None:
        """Initialize the profiler with the given options."""
        self.set_options(self.options)

    def set_options(self, options: Options) -> None:
        """Set the options for the profiler."""
        self.options = options
        # simulation related
        self.sim_compute = options.sim_compute
        self.sim_sliding_window = options.sim_sliding_window
        self.sim_num_key_value_heads = options.sim_num_key_value_heads

        self.sim_data_type = options.sim_data_type

        self.sim_data_type_bytes = {
            "int4": 0.5,
            "int8": 1.0,
            "float16": 2.0,
            "bfloat16": 2.0,
            "float32": 4.0,
        }.get(self.sim_data_type) or self.sim_data_type_bytes

        self.simulator = self._create_arch_simulator() if options.simulation else None

    def _create_arch_simulator(self) -> Simulator:
        return Simulator(
            data_type_bytes=self.sim_data_type_bytes,
            sliding_window=self.sim_sliding_window,
            num_key_value_heads=self.sim_num_key_value_heads,
            verbose=self.options.sim_verbose,
        )

    def _print_layers_model(self) -> None:
        print("##### Layers of Model in order of creation #####")
        print(
            "Layer ID (creation order), Context, Function, Dimensions (rows x columns), "
            "times executed, avg. execution time (ms)"
        )
        for layer in self.layers.values():
            print(
                f"{layer.id}, {layer.context}, {layer.name}, "
                f"({layer.dim_in}x{layer.dim_out}), {layer.exec_nums}, "
                f"{layer.exec_time / self.n_executions / 1e6}"
            )

    def _print_functions_model(self) -> None:
        print("##### Functions called by the Model in order of calling #####")
        print(
            "Function name, Context, Dimensions in (columns), Dimensions out (columns), "
            "times executed, avg. execution time (ms)"
        )
        for name, func in self.functions.items():
            print(
                f"{name}, {func.context}, ({func.dim_in}), ({func.dim_out}), "
                f"{func.exec_nums}, {func.exec_time / 1e6}"
            )

    def _print_log_summary(
        self, *, show_summarization: bool = False, show_all: bool = False
    ) -> None:
        phase = "Generation"
        if show_summarization:
            phase = "Summarization"
        if show_all:
            phase = "All (SUM and GEN)"
        print("#####", phase, "Execution summary #####")
        name_ctxt = []
        summary_time = OrderedDict()
        summary_perf = OrderedDict()
        summary_energy = OrderedDict()
        summary_transfer_bytes = OrderedDict()
        summary_nexec = OrderedDict()
        input_shapes = OrderedDict()
        weights_shapes = OrderedDict()
        output_shapes = OrderedDict()
        for log in self.log:
            if not show_all and not show_summarization and log.summarization:
                continue
            if not show_all and show_summarization and not log.summarization:
                continue
            ctxt = log.name + ":" + log.context
            if ctxt not in summary_time:
                name_ctxt.append(ctxt)
            summary_nexec[ctxt] = 1 + summary_nexec.get(ctxt, 0)
            summary_time[ctxt] = log.exec_time_ms + summary_time.get(ctxt, 0)
            summary_energy[ctxt] = add_dictionaries(summary_energy.get(ctxt, {}), log.energy)
            summary_perf[ctxt] = add_dictionaries(summary_perf.get(ctxt, {}), log.performance)
            summary_transfer_bytes[ctxt] = add_dictionaries(
                summary_transfer_bytes.get(ctxt, {}), log.transfer_bytes
            )

            input_shapes[ctxt] = "(" + ":".join([str(x) for x in log.input]) + ")"
            weights_shapes[ctxt] = "(" + ":".join([str(x) for x in log.weights]) + ")"
            output_shapes[ctxt] = "(" + ":".join([str(x) for x in log.output]) + ")"

        executed_times = 1 if show_summarization else (self.n_executions - 1)
        print(
            "Function: Context: input shape: weights shape: output shape:"
            "time(s):H2C(ms):C2H(ms):compute(ms):mem_transfer(ms):kv_load(ms)"
            "host_to_device(mJ):device_to_host(mJ):main_mem(mJ):compute(mJ)"
        )
        for key in name_ctxt:
            perf_values = [
                str(summary_perf[key].get(perf_key, 0) / 1e6 / executed_times)
                for perf_key in (
                    "host_to_device",
                    "device_to_host",
                    "compute",
                    "mem_transfer",
                    "kv_load",
                )
            ]
            perf_string = ":".join(perf_values)

            energy_values = [
                str(summary_energy[key].get(ene_key, 0) / 1e6 / executed_times)
                for ene_key in (
                    "host_to_device",
                    "device_to_host",
                    "main_mem",
                    "compute",
                )
            ]
            energy_string = ":".join(energy_values)

            print(
                key,
                input_shapes[key],
                weights_shapes[key],
                output_shapes[key],
                (summary_time[key] / executed_times),
                perf_string,
                energy_string,
            )

        total_time_explained = sum(summary_time.values())
        total_percentage_explained = 0
        for key in name_ctxt:
            print(
                f"{key} explains {(summary_time[key] / total_time_explained) * 100} % of the total "
                f"inference time (num. executions: {summary_nexec[key]}) "
                f"average time: {summary_time[key] / summary_nexec[key]} (ms)",
            )
            total_percentage_explained += (summary_time[key] / total_time_explained) * 100
        print("Profiler captures", total_percentage_explained, "% of the total execution")
        print("Profiler captures", total_time_explained, "ms of the total execution")
        print(summary_time)

    def _print_log(self) -> None:
        print("##### Execution log #####")
        print("Start time, exec time, Function, Context, input shape, weights shape, output shape")
        for log in self.log:
            input_shape = "(" + ",".join([str(x) for x in log.input]) + ")"
            weights_shape = "(" + ",".join([str(x) for x in log.weights]) + ")"
            output_shape = "(" + ",".join([str(x) for x in log.output]) + ")"
            print(
                log.start_time / 1e6,
                log.exec_time_ms,
                log.id,
                log.name,
                log.context,
                input_shape,
                weights_shape,
                output_shape,
            )

    def _update_inference_perf(
        self, step_perf: dict[str, float], *, summarization_phase: bool
    ) -> None:
        if summarization_phase:
            self.sum_perf = add_dictionaries(self.sum_perf, step_perf)
        else:
            self.gen_perf = add_dictionaries(self.gen_perf, step_perf)

    def _update_inference_energy(
        self, step_energy: dict[str, float], *, summarization_phase: bool
    ) -> None:
        if summarization_phase:
            self.sum_energy = add_dictionaries(self.sum_energy, step_energy)
        else:
            self.gen_energy = add_dictionaries(self.gen_energy, step_energy)

    def _update_inference_transfer_bytes(
        self, step_transfer_bytes: dict[str, float], *, summarization_phase: bool
    ) -> None:
        if summarization_phase:
            self.sum_transfer_bytes = add_dictionaries(self.sum_transfer_bytes, step_transfer_bytes)
        else:
            self.gen_transfer_bytes = add_dictionaries(self.gen_transfer_bytes, step_transfer_bytes)

    def start(
        self,
        layer_mapping: dict[str, str] | None = None,
        layer_attn_ctxt: str = "",
        last_layer: str = "lm_head",
        batch_size: int = 1,
        moe_end: str = "",
        experts_per_token: int = 2,
    ) -> None:
        """Start the profiler."""
        if layer_mapping is None:
            layer_mapping = {}

        self.start_inference = time.time_ns()
        self.n_executions = 0
        self.inference_time = 0
        self.summarization_time = 0
        self.sum_perf = {}
        self.gen_perf = {}
        self.sum_energy = {}
        self.gen_energy = {}
        self.sum_transfer_bytes = {}
        self.gen_transfer_bytes = {}

        self.last_layer = last_layer
        self.batch_size = batch_size

        self.layers_start = {}
        self.layers_end = {}
        self.log = []

        if self.simulator:
            self.simulator.map_layers(
                layer_mapping,
                layer_attn_ctxt=layer_attn_ctxt,
                moe_end=moe_end,
                experts_per_token=experts_per_token,
            )

    def end(self) -> None:
        """End the profiler."""
        if self.simulator:
            step_time, step_perf, step_energy, step_transfer_bytes = self.simulator.simulate_end(
                self.forward_input_shape, generated_tokens=(self.n_executions)
            )
            self.inference_time += step_time
            summarization_phase = self.simulator.sum
            self._update_inference_perf(step_perf, summarization_phase=summarization_phase)
            self._update_inference_energy(step_energy, summarization_phase=summarization_phase)
            self._update_inference_transfer_bytes(
                step_transfer_bytes, summarization_phase=summarization_phase
            )
        else:
            self.inference_time = time.time_ns() - self.start_inference

        inference_time_sec = self.inference_time / 1e9
        sum_energy_mj = 0
        gen_energy_mj = 0
        sum_time_s = self.summarization_time / 1e9
        gen_time_s = inference_time_sec - sum_time_s
        gen_n_executions = self.n_executions - 1

        print("##### UPMEM PROFILER OUTPUT #####")
        print(
            f"Total time (SUM + GEN): {inference_time_sec} s, "
            f"with data type: {self.sim_data_type}, "
            f"batch size: {self.batch_size}"
        )
        print(
            f"Generated tokens: {gen_n_executions * self.batch_size} "
            f"in {gen_time_s} s, "
            f"with tokens/s: {(gen_n_executions * self.batch_size) / gen_time_s}"
        )
        print(
            f"Summarization step took: {sum_time_s} s, "
            f"weight in the execution: SUM: {sum_time_s / inference_time_sec}%, "
            f"GEN: {gen_time_s / inference_time_sec}%"
        )

        if self.simulator:
            print("SUMMARIZATION summary")
            for key, transfer in self.sum_transfer_bytes.items():
                print(f"Transferred data in {key}: {transfer / 1e6} MB")
            for key, energy in self.sum_energy.items():
                energy_mj = energy / 1e9
                print(f"Energy in {key}: {energy_mj} mJ")
                sum_energy_mj += energy / 1e9
            print(f"Energy: {sum_energy_mj} mJ")
            print(f"Power: {sum_energy_mj / 1e3 / sum_time_s} W")

            if gen_n_executions > 0:
                print("GENERATION summary")
                for key, transfer in self.gen_transfer_bytes.items():
                    print(
                        f"Transferred data in {key}: {transfer / 1e6} MB, "
                        f"MB/token: {transfer / 1e6 / self.n_executions / self.batch_size}"
                    )
                for key, energy in self.gen_energy.items():
                    energy_mj = energy / 1e9
                    print(
                        f"Energy in {key}: {energy_mj} mJ, "
                        f"mJ/token: {energy_mj / gen_n_executions / self.batch_size}"
                    )
                    gen_energy_mj += energy_mj
                print(
                    f"Energy: {gen_energy_mj} mJ, "
                    f"mJ/token: {gen_energy_mj / gen_n_executions / self.batch_size}"
                )
                print(f"Power: {gen_energy_mj / 1e3 / gen_time_s} W")

            print("Execution time breakdown (ms / %)")
            print("SUMMARIZATION phase")
            for perf_key in [
                "host_to_device",
                "device_to_host",
                "compute",
                "mem_transfer",
                "kv_load",
            ]:
                perf_value = self.sum_perf.get(perf_key, 0)
                print(perf_key, (perf_value / 1e6), "(ms)", perf_value / 1e9 / sum_time_s)

            if gen_n_executions > 0:
                print("GENERATION phase")
                for perf_key in [
                    "host_to_device",
                    "device_to_host",
                    "compute",
                    "mem_transfer",
                    "kv_load",
                ]:
                    perf_value = self.gen_perf.get(perf_key, 0)
                    print(f"{perf_key}: {perf_value / 1e6} ms, {perf_value / 1e9 / gen_time_s}")

        if self.options.report_layers:
            self._print_layers_model()

        if self.options.report_functions:
            self._print_functions_model()

        if self.options.print_log:
            self._print_log()

        if self.options.print_log_summary:
            self._print_log_summary()
            self._print_log_summary(show_summarization=True)
            self._print_log_summary(show_all=True)

        print("##### END UPMEM PROFILER OUTPUT #####")

    def add(self, layer: torch.nn.Module, context: str) -> None:
        """Add a layer to the profiler."""
        layer_type = type(layer)
        dim_in, dim_out = next(
            (
                dim_func(layer)
                for key, dim_func in self.layer_dimensions.items()
                if issubclass(layer_type, key)
            ),
            (None, None),
        )

        if dim_in is None or dim_out is None:
            print(f"Layer: {layer_type} not supported")
            sys.exit()

        name_layer = layer_type.__name__

        self.layers[layer] = LayerProfile(
            self.n_layers,
            name_layer,
            self.n_layers,
            context,
            dim_in,
            dim_out,
            obj=layer,
        )
        self.n_layers += 1

    def forward_start(self, input_shape: torch.Size) -> None:
        """Start profiling a forward pass."""
        self.forward_input_shape = input_shape

        if self.simulator:
            self.forward_time_start = self.inference_time
        else:
            self.forward_time_start = time.time_ns()

    def forward_end(
        self, output_shape: torch.Size, context: str, layer_obj: torch.nn.Module
    ) -> None:
        """End profiling a forward pass."""
        self.forward_time_end = time.time_ns()

        weights_shape = torch.Size([self.layers[layer_obj].dim_in, self.layers[layer_obj].dim_out])

        cur_exec_time = self.forward_time_end - self.forward_time_start
        performance = {}
        energy = {}
        relative_start_time = self.forward_time_start - self.start_inference

        if self.simulator:
            if any(isinstance(layer_obj, layer_t) for layer_t in self.functional_layers):
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_function(
                        self.layers[layer_obj],
                        context,
                        output_shape,
                        self.layers[layer_obj].dim_out,
                    )
                )
            else:
                cur_exec_time, performance, energy, transfer_bytes = self.simulator.simulate_layer(
                    self.layers[layer_obj],
                    self.forward_input_shape,
                    layer_obj,
                    weights_shape,
                    output_shape,
                )  # or context?
            self.inference_time += cur_exec_time
            summarization_phase = self.simulator.sum
            self._update_inference_perf(performance, summarization_phase=summarization_phase)
            self._update_inference_energy(energy, summarization_phase=summarization_phase)
            self._update_inference_transfer_bytes(
                transfer_bytes, summarization_phase=summarization_phase
            )

            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers[layer_obj].energy = add_dictionaries(self.layers[layer_obj].energy, energy)
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.inference_time
            relative_start_time = self.forward_time_start
        else:
            self.inference_time += cur_exec_time
            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.forward_time_end
            transfer_bytes = {}

        self.layers[layer_obj].exec_nums += 1

        summarization_phase = self.simulator.sum if self.simulator else False

        cur_logging = LayerLog(
            self.layers[layer_obj].id,
            self.layers[layer_obj].name,
            self.layers[layer_obj].context,
            summarization_phase,
            relative_start_time / 1e6,
            self.forward_input_shape,
            weights_shape,
            output_shape,
            cur_exec_time / 1e6,
            performance,
            energy,
            transfer_bytes,
        )

        self.log.append(cur_logging)
        if self.layers[layer_obj].context == self.last_layer:
            if self.simulator and not self.simulator.sum:
                self.simulator.sum_size += 1
            if self.n_executions == 0:
                if self.simulator:
                    self.simulator.start_gen()
                    self.simulator.sum_size = output_shape[-2] if (len(output_shape) > 1) else 1
                self.summarization_time = self.inference_time
            self.n_executions += 1
            print(f"New token generated ({self.n_executions})", end="\r")

    def forward_func_start(self, input_shape: torch.Size) -> None:
        """Start profiling a function call."""
        self.func_input_shape = input_shape

        self.start_func = self.inference_time if self.simulator else time.time_ns()

    def forward_func_end(self, function: Callable, context: str, output_shape: torch.Size) -> None:
        """End profiling a function call."""
        self.end_func = time.time_ns()

        func_profile = self.functions.get(
            function.__name__,
            LayerProfile(
                0, function.__name__, 0, context, self.func_input_shape[-1], output_shape[-1]
            ),
        )

        cur_exec_time = self.end_func - self.start_func
        performance = {}
        energy = {}

        if self.simulator:
            cur_exec_time, performance, energy, transfer_bytes = self.simulator.simulate_function(
                function, context, self.func_input_shape, output_shape
            )
            self.inference_time += cur_exec_time
            summarization_phase = self.simulator.sum
            self._update_inference_perf(performance, summarization_phase=summarization_phase)
            self._update_inference_energy(energy, summarization_phase=summarization_phase)
            self._update_inference_transfer_bytes(
                transfer_bytes, summarization_phase=summarization_phase
            )
        else:
            transfer_bytes = {}

        relative_exec_time = self.start_func

        summarization_phase = self.simulator.sum if self.simulator else False
        cur_logging = LayerLog(
            0,  # functions have ID set to 0
            function.__name__,
            context,
            summarization_phase,
            relative_exec_time / 1e6,
            self.func_input_shape,
            output_shape,
            output_shape,
            cur_exec_time / 1e6,
            performance,
            energy,
            transfer_bytes,
        )

        self.log.append(cur_logging)

        func_profile.exec_nums += 1
        func_profile.exec_time += cur_exec_time
        self.functions[function.__name__] = func_profile
