#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements all profiling related classes and functions

import sys
import time
from collections import OrderedDict
from typing import Mapping, Callable, Tuple

import torch
from torch.nn import Linear, SiLU, LayerNorm, Embedding, Dropout, Softmax, Conv2d
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.activations import NewGELUActivation
from transformers.pytorch_utils import Conv1D

from upmem_llm_framework.simulator import Simulator
from upmem_llm_framework.utils import add_dictionaries


class layer_profile:
    def __init__(self, uniq_id, name, n_layer, context, dim_in, dim_out, obj=None):
        self.id = uniq_id
        self.name = name
        self.n_layer = n_layer
        self.context = context
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.exec_time = 0
        self.exec_nums = 0
        self.energy = {}
        self.obj = obj


class layer_log:
    def __init__(
        self,
        uniq_id,
        name,
        context,
        summarization,
        start_time,
        input,
        weights,
        output,
        exec_time_ms,
        performance,
        energy_pj,
        transfer_bytes,
    ):
        self.id = uniq_id
        self.name = name
        self.context = context
        self.summarization = summarization
        self.start_time = start_time
        self.input = input
        self.weights = weights
        self.output = output
        self.exec_time_ms = exec_time_ms
        self.performance = performance
        self.energy = energy_pj
        self.transfer_bytes = transfer_bytes


class UPM_Profiler:
    layer_dimensions: Mapping[type, Callable[[torch.nn.Module], Tuple[int, int]]] = {
        Linear: (lambda l: (l.in_features, l.out_features)),
        NewGELUActivation: (lambda l: (1, 1)),
        SiLU: (lambda l: (1, 1)),
        LlamaRMSNorm: (lambda l: (l.weight.size()[0], l.weight.size()[0])),
        LlamaRotaryEmbedding: (lambda l: (1, 1)),
        LayerNorm: (lambda l: (1, 1)),
        Embedding: (lambda l: (1, 1)),
        Dropout: (lambda l: (1, 1)),
        Softmax: (lambda l: (l.dim, l.dim)),
        Conv1D: (lambda l: (l.weight.shape[0], l.weight.shape[1])),
        Conv2d: (lambda l: (l.kernel_size[0], l.kernel_size[1])),
    }

    functional_layers = {LlamaRMSNorm, SiLU}

    def __init__(self, options):
        self.n_layers = 0
        self.n_executions = 0
        self.layers = {}
        self.functions = {}

        self.options = options
        self.sim_compute = False
        self.sim_sliding_window = -1
        self.sim_num_key_value_heads = 8
        self.sim_data_type = "float16"
        self.sim_data_type_bytes = 2.0
        self.simulator = None
        self.set_options(options)

        self.start_inference = time.time_ns()
        self.inference_time = 0
        self.summarization_time = 0
        self.sum_perf = {}
        self.gen_perf = {}
        self.sum_energy = {}
        self.gen_energy = {}
        self.sum_transfer_bytes = {}
        self.gen_transfer_bytes = {}

        self.last_layer = "lm_head"
        self.batch_size = 1

        self.layers_start = {}
        self.layers_end = {}
        self.log = []

        self.forward_input_shape = None
        self.forward_time_start = 0
        self.forward_time_end = 0

        self.func_input_shape = None
        self.start_func = 0
        self.end_func = 0

    def set_options(self, options):
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
        }.get(self.sim_data_type)

        self.simulator = self.create_arch_simulator() if options.simulation else None

    def create_arch_simulator(self):
        return Simulator(
            data_type_bytes=self.sim_data_type_bytes,
            sliding_window=self.sim_sliding_window,
            num_key_value_heads=self.sim_num_key_value_heads,
            verbose=self.options.sim_verbose,
        )

    def print_layers_model(self):
        print("##### Layers of Model in order of creation #####")
        print(
            "Layer ID (creation order), Context, Function, Dimensions (rows x columns), "
            "times executed, avg. execution time (ms)"
        )
        for layer in self.layers.values():
            print(
                f"{str(layer.id)},"
                f"{layer.context},"
                f"{layer.name},"
                f"({layer.dim_in}x{layer.dim_out}),"
                f"{layer.exec_nums},"
                f"{layer.exec_time / self.n_executions / 1e6}"
            )

    def print_functions_model(self):
        print("##### Functions called by the Model in order of calling #####")
        print(
            "Function name, Context, Dimensions in (columns), Dimensions out (columns), "
            "times executed, avg. execution time (ms)"
        )
        for name, func in self.functions.items():
            print(
                f"{str(name)},"
                f"{func.context},"
                f"({func.dim_in}),"
                f"({func.dim_out}),"
                f"{func.exec_nums},"
                f"{func.exec_time / 1e6}"
            )

    def print_log_summary(self, show_summarization=False, show_all=False):
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
            if not ctxt in summary_time.keys():
                name_ctxt.append(ctxt)
            summary_nexec[ctxt] = 1 + summary_nexec.get(ctxt, 0)
            summary_time[ctxt] = log.exec_time_ms + summary_time.get(ctxt, 0)
            summary_energy[ctxt] = add_dictionaries(
                summary_energy.get(ctxt, {}), log.energy
            )
            summary_perf[ctxt] = add_dictionaries(
                summary_perf.get(ctxt, {}), log.performance
            )
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

            perf_values = []
            for perf_key in [
                "host_to_device",
                "device_to_host",
                "compute",
                "mem_transfer",
                "kv_load",
            ]:
                perf_values.append(
                    str(summary_perf[key].get(perf_key, 0) / 1e6 / executed_times)
                )
            perf_string = ":".join(perf_values)

            energy_values = []
            for ene_key in ["host_to_device", "device_to_host", "main_mem", "compute"]:
                energy_values.append(
                    str(summary_energy[key].get(ene_key, 0) / 1e6 / executed_times)
                )
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
                key,
                "explains",
                (summary_time[key] / total_time_explained) * 100,
                "% of the total inference time (num. executions:",
                summary_nexec[key],
                ") average time:",
                summary_time[key] / summary_nexec[key],
                "(ms)",
            )
            total_percentage_explained += (
                summary_time[key] / total_time_explained
            ) * 100
        print(
            "Profiler captures", total_percentage_explained, "% of the total execution"
        )
        print("Profiler captures", total_time_explained, "ms of the total execution")
        print(summary_time)

    def print_log(self):
        print("##### Execution log #####")
        print(
            "Start time, exec time, Function, Context, input shape, weights shape, output shape"
        )
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

    def update_inference_perf(self, step_perf):
        if self.simulator.sum:
            for key in step_perf.keys():
                self.sum_perf[key] = self.sum_perf.get(key, 0) + step_perf[key]
        else:
            for key in step_perf.keys():
                self.gen_perf[key] = self.gen_perf.get(key, 0) + step_perf[key]

    def update_inference_energy(self, step_energy):
        if self.simulator.sum:
            for key in step_energy.keys():
                self.sum_energy[key] = self.sum_energy.get(key, 0) + step_energy[key]
        else:
            for key in step_energy.keys():
                self.gen_energy[key] = self.gen_energy.get(key, 0) + step_energy[key]

    def update_inference_transfer_bytes(self, step_transfer_bytes):
        if self.simulator.sum:
            for key in step_transfer_bytes.keys():
                self.sum_transfer_bytes[key] = (
                    self.sum_transfer_bytes.get(key, 0) + step_transfer_bytes[key]
                )
        else:
            for key in step_transfer_bytes.keys():
                self.gen_transfer_bytes[key] = (
                    self.gen_transfer_bytes.get(key, 0) + step_transfer_bytes[key]
                )

    def start(
        self,
        layer_mapping=None,
        layer_attn_ctxt="",
        last_layer="lm_head",
        batch_size=1,
        moe_end="",
        experts_per_token=2,
    ):
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

    def end(self):
        if self.simulator:
            step_time, step_perf, step_energy, step_transfer_bytes = (
                self.simulator.simulate_end(
                    self.forward_input_shape, generated_tokens=(self.n_executions)
                )
            )
            self.inference_time += step_time
            self.update_inference_perf(step_perf)
            self.update_inference_energy(step_energy)
            self.update_inference_transfer_bytes(step_transfer_bytes)
        else:
            self.inference_time = time.time_ns() - self.start_inference

        inference_time_sec = self.inference_time / 1e9
        sum_energy_mJ = 0
        gen_energy_mJ = 0
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
                sum_energy_mJ += energy / 1e9
            print(f"Energy: {sum_energy_mJ} mJ")
            print(f"Power: {sum_energy_mJ / 1e3 / sum_time_s} W")

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
                    gen_energy_mJ += energy_mj
                print(
                    f"Energy: {gen_energy_mJ} mJ, "
                    f"mJ/token: {gen_energy_mJ / gen_n_executions / self.batch_size}"
                )
                print(f"Power: {gen_energy_mJ / 1e3 / gen_time_s} W")

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
                print(
                    perf_key, (perf_value / 1e6), "(ms)", perf_value / 1e9 / sum_time_s
                )

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
                    print(
                        f"{perf_key}: {perf_value / 1e6} ms, {perf_value / 1e9 / gen_time_s}"
                    )

        if self.options.report_layers:
            self.print_layers_model()

        if self.options.report_functions:
            self.print_functions_model()

        if self.options.print_log:
            self.print_log()

        if self.options.print_log_summary:
            self.print_log_summary()
            self.print_log_summary(show_summarization=True)
            self.print_log_summary(show_all=True)

        print("##### END UPMEM PROFILER OUTPUT #####")

    def add(self, layer, context):
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

        self.layers[layer] = layer_profile(
            self.n_layers,
            name_layer,
            self.n_layers,
            context,
            dim_in,
            dim_out,
            obj=layer,
        )
        self.n_layers += 1

    def forward_start(self, input_shape):
        self.forward_input_shape = input_shape

        if self.simulator:
            self.forward_time_start = self.inference_time
        else:
            self.forward_time_start = time.time_ns()

    def forward_end(self, output_shape, context, layer_obj=None):
        self.forward_time_end = time.time_ns()

        weights_shape = torch.Size(
            [self.layers[layer_obj].dim_in, self.layers[layer_obj].dim_out]
        )

        cur_exec_time = self.forward_time_end - self.forward_time_start
        performance = {}
        energy = {}
        relative_start_time = self.forward_time_start - self.start_inference

        if self.simulator:
            name = self.layers[layer_obj].name
            if any(
                isinstance(layer_obj, layer_t) for layer_t in self.functional_layers
            ):
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_function(
                        self.layers[layer_obj],
                        context,
                        output_shape,
                        self.layers[layer_obj].dim_out,
                    )
                )
            else:
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_layer(
                        self.layers[layer_obj],
                        self.forward_input_shape,
                        layer_obj,
                        weights_shape,
                        output_shape,
                    )
                )  # or context?
            self.inference_time += cur_exec_time
            self.update_inference_perf(performance)
            self.update_inference_energy(energy)
            self.update_inference_transfer_bytes(transfer_bytes)

            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers[layer_obj].energy = add_dictionaries(
                self.layers[layer_obj].energy, energy
            )
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.inference_time
            relative_start_time = self.forward_time_start
        else:
            self.inference_time += cur_exec_time
            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.forward_time_end
            transfer_bytes = 0

        self.layers[layer_obj].exec_nums += 1

        summarization_phase = self.simulator.sum if self.simulator else False

        cur_logging = layer_log(
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
                    self.simulator.sum_size = (
                        output_shape[-2] if (len(output_shape) > 1) else 1
                    )
                self.summarization_time = self.inference_time
            self.n_executions += 1
            print(f"New token generated ({self.n_executions})", end="\r")

    def forward_func_start(self, name, context, input_shape):
        self.func_input_shape = input_shape

        self.start_func = time.time_ns()
        if self.simulator:
            self.start_func = self.inference_time

    def forward_func_end(self, function, context, output_shape):
        self.end_func = time.time_ns()

        func_profile = self.functions.get(
            function.__name__,
            layer_profile(
                0, function.__name__, 0, context, self.func_input_shape[-1], output_shape[-1]
            ),
        )

        cur_exec_time = self.end_func - self.start_func
        performance = {}
        energy = {}
        relative_time = self.start_func - self.start_inference

        if self.simulator:
            cur_exec_time, performance, energy, transfer_bytes = (
                self.simulator.simulate_function(
                    function, context, self.func_input_shape, output_shape
                )
            )
            self.inference_time += cur_exec_time
            self.update_inference_perf(performance)
            self.update_inference_energy(energy)
            self.update_inference_transfer_bytes(transfer_bytes)
        else:
            transfer_bytes = 0

        relative_exec_time = self.start_func

        summarization_phase = self.simulator.sum if self.simulator else False
        cur_logging = layer_log(
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
