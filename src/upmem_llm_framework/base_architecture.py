"""Implements Base_architecture class.

This class contains a default implementation of the following functions:
- `adjust_for_quantization`: scales up/down the TFLOPs depending on the quantization choosen
- `get_tflops`: returns the TFLOPs required in a MxM
- `get_moved_data_bytes`: returns the required bytes to move in order to do an operation
- `load_data`: models loading the KV cache
- `host_transfer`: simulates a data transfer with host in any direction
- `compute_ns`: simulates the computation of a MxM
- `compute_scaled_dot_product_ns`: simulates the computation of function scaled_dot_product where,
  `usually`, attention computation occurs
- `compute_matmul_ns`: simulates the computation of a matmul for self-attention
- `compute_activation_ns`: simulates an activation layer
- `compute_RMSNorm_ns`: simulates a RMSNorm layer
- `compute_softmax_ns`: simulates a softmax operation

Note that all simulations returns compute_time_ns, performance_dict, energy_dict:
- `compute_time_ns`: the simulated time in ns,
- `performance_dict`: dictionary containing the simulated time in ns for each operation simulated,
- `energy_dict`: dictionary containing the simulated energy in pJ for each operation simulated.
"""

import math
import typing
import warnings
from dataclasses import dataclass

import torch

from upmem_llm_framework.utils import add_dictionaries


@dataclass
class BaseArchitecture:
    """Architectures class.

    This class contains the implementation of the functions
    that are used to simulate an architecture.
    """

    name: str = ""
    # Compute capabilities
    tflops: float = 1
    pj_per_tflop: float = 1
    # Interface with HOST
    host_to_device_bw_GBs: float = 1
    device_to_host_bw_GBs: float = 1
    host_to_device_pj_per_bit: float = 25
    device_to_host_pj_per_bit: float = 25
    # Device memory (shared memory like)
    mem_bw_GBs: float = 1
    mem_pj_per_bit: float = 1

    data_type_bytes: float = 2  # float16
    # 3000 cycles per row of 2048 elements --> 1.4 cycles / element
    # assuming 1 GHz, 1.5 ns / element, parallelized accross 4 chips -> 0.37
    softmax_ns_per_element: float = 0.4  # ns, considering it cycles in 1GHz config
    SiLU_ns_per_element: float = 0.6  # ns, softmax * 1.5
    # (empiric number based on execution of Llama2-7b)
    RMSNorm_ns_per_element: float = 1.1  # ns, softmax * 2.6
    # (empiric number based on execution of Llama2-7b)
    # 3000 cycles per row of 2048 elements with 5 TFLOPs of computing power
    # assuming 1 GHz, 0,000003 s --> 3MOPS per row of 2048 --> 1.5kOPS per element
    misc_tflops_per_element: float = 1500 / 1e12
    sliding_window: int | None = None
    num_key_value_heads: int | None = None
    verbose: bool = False

    def load_spec(self, name: str, spec: dict) -> None:
        """Load accelerator specification from a dictionary."""
        self.name = name
        for key, value in spec.items():
            if key == "tflops_int4":
                continue
            if not hasattr(self, key):
                msg = f"Warning: {key} is not a valid attribute for {self.__class__}"
                raise ValueError(msg)
            setattr(self, key, value)
        if "tflops_int4" in spec and self.data_type_bytes == 0.5:
            self.tflops = spec["tflops_int4"]

    def adjust_for_quantization(self) -> None:
        """Adjust the energy consumption based on the quantization.

        Defined TFLOPS are defined for float16,
        assumes that performance is doubled if data type is demoted.
        """
        ratio = 2 / self.data_type_bytes  # Assume pj_per_tflop corresponds to float16
        self.pj_per_tflop = self.pj_per_tflop / ratio
        # self.tflops = self.tflops * (2 / self.data_type_bytes)

        # self.softmax_ns_per_element = self.softmax_ns_per_element / ratio
        # self.RMSNorm_ns_per_element = self.RMSNorm_ns_per_element / ratio
        # self.SiLU_ns_per_element = self.SiLU_ns_per_element / ratio

    def _get_tflops_linear(self, input_shape: torch.Size, weight_shape: torch.Size) -> float:
        out_features = weight_shape[0]
        return input_shape.numel() * out_features * 2 / 1e12

    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    def _get_tflops_conv2d(
        self, input_shape: torch.Size, layer: torch.nn.Conv2d, weight_shape: torch.Size
    ) -> float:
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_channels = input_shape[-3] if (len(input_shape) > 2) else 1
        n_height = input_shape[-2] if (len(input_shape) > 1) else 1
        n_width = input_shape[-1]

        stride = layer.stride
        padding = layer.padding

        if isinstance(padding, str):
            err = f"Padding type {padding} not implemented. Please use int or tuple."
            raise NotImplementedError(err)

        n_height = n_height + 2 * (padding[0] if isinstance(padding, tuple) else padding)
        n_width = n_width + 2 * (padding[1] if isinstance(padding, tuple) else padding)

        # Example of how many times a kernel is applied depending on stride:
        # | 0 | 1 | 2 | 3 | 4 | 5 |
        # Stride 1:
        # 00001111
        #     11112222
        #         22223333
        #             33334444
        #                 44445555
        # Stride 2:
        # 00001111
        #         22223333
        #                 44445555

        width_times = math.ceil(
            (n_width - 1) / (stride[1] if isinstance(stride, tuple) else stride)
        )
        height_times = math.ceil(
            (n_width - 1) / (stride[0] if isinstance(stride, tuple) else stride)
        )

        # TFLOPS when applying once the kernel
        tflops_kernel = (2 * batch_size * n_channels * weight_shape[1] * weight_shape[0]) / 1e12

        return tflops_kernel * width_times * height_times

    def _get_tflops_layernorm(self, input_shape: torch.Size) -> float:
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        return batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element

    # TODO: see DeepSpeed implementation
    # def _attn_flops_compute

    def _get_tflops(
        self, input_shape: torch.Size, layer: torch.nn.Module, weight_shape: torch.Size
    ) -> float:
        if isinstance(layer, torch.nn.Conv2d):
            tflops = self._get_tflops_conv2d(
                input_shape, typing.cast("torch.nn.Conv2d", layer), weight_shape
            )
        elif isinstance(layer, torch.nn.LayerNorm):
            tflops = self._get_tflops_layernorm(input_shape)
        else:
            # Treat everything else as Linear
            tflops = self._get_tflops_linear(input_shape, weight_shape)
            if not isinstance(layer, torch.nn.Linear):
                warnings.warn(
                    f"get_tflops not defined for layer: {type(layer)}. Using Linear as default.",
                    stacklevel=2,
                )

        return tflops

    def _get_moved_data_bytes(
        self,
        input_shape: torch.Size,
        weight_shape: torch.Size,
        *,
        load_input: bool = False,
        load_weight: bool = True,
    ) -> float:
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        weight_size = weight_shape[1] * weight_shape[0] if load_weight else 0
        input_size = batch_size * n_heads * n_rows * n_columns if load_input else 0
        return self.data_type_bytes * (weight_size + input_size)

    def load_data(
        self, input_shape: torch.Size
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for KV cache load."""
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        # TODO: read the data type from the model
        # data_size_bytes = self.data_type_bytes * (
        #     batch_size * n_rows * n_heads * n_columns
        # )

        # Hardcoded to FP16
        data_size_bytes = 2 * (batch_size * n_rows * n_heads * n_columns)
        # B / (GB/s) --> s/G --> ns
        transfer_time_ns = data_size_bytes / self.mem_bw_GBs

        performance = {"kv_load": transfer_time_ns}
        # GB/s * time --> GB * pJ/bit --> J
        energy = {"main_mem": data_size_bytes * 8 * self.mem_pj_per_bit}

        if self.verbose:
            print(
                f"Load time for input_shape: {input_shape} = {transfer_time_ns} (ns) with "
                f"{energy} pj {performance}"
            )

        return transfer_time_ns, performance, energy

    def host_transfer(
        self, input_shape: torch.Size, *, direction: str = "to_device", generated_tokens: int = 1
    ) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
        """Compute metrics for host <-> device transfer."""
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        bandwidth = (
            self.host_to_device_bw_GBs if direction == "to_device" else self.device_to_host_bw_GBs
        )

        data_size_bytes = self.data_type_bytes * (
            batch_size * n_heads * n_rows * n_columns * generated_tokens
        )
        # B / (GB/s) --> s / G --> ns
        transfer_time_ns = data_size_bytes / bandwidth

        name_op = "host_to_device" if direction == "to_device" else "device_to_host"

        performance = {name_op: transfer_time_ns}
        moved_data = {name_op: data_size_bytes}

        ddr_pj_per_bit = (
            self.host_to_device_pj_per_bit
            if direction == "to_device"
            else self.device_to_host_pj_per_bit
        )
        energy = {name_op: data_size_bytes * 8 * ddr_pj_per_bit}

        if self.verbose:
            print(
                f"Transfer time for input_shape: {input_shape} = {transfer_time_ns} (ns) "
                f"with {energy} pj perf: {performance} data in bytes: {moved_data}"
            )

        return transfer_time_ns, performance, energy, moved_data

    def compute_ns(
        self,
        input_shape: torch.Size,
        layer_obj: torch.nn.Module,
        weight_shape: torch.Size,
        *,
        load_input: bool = False,
        load_weight: bool = True,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for a layer."""
        tflops = self._get_tflops(input_shape, layer_obj, weight_shape)

        data_size_bytes = self._get_moved_data_bytes(
            input_shape, weight_shape, load_input=load_input, load_weight=load_weight
        )

        compute_time_ns = (tflops / self.tflops) * 1e9
        transfer_time_ns = data_size_bytes / self.mem_bw_GBs
        real_time_ns = max(compute_time_ns, transfer_time_ns)

        performance = {
            "compute": compute_time_ns,
            "mem_transfer": transfer_time_ns,
        }

        energy = {
            "compute": tflops * self.pj_per_tflop,
            "main_mem": data_size_bytes * 8 * self.mem_pj_per_bit,
        }

        if self.verbose:
            print(
                f"Computing {input_shape} x {weight_shape} with TFLOPS: {tflops} "
                f"with {data_size_bytes} bytes"
            )
            print(
                f"takes {real_time_ns} ns with {compute_time_ns} in compute and {transfer_time_ns} "
                f"in loading data"
            )
            print(
                f"and consumes {energy['compute']} pJ for compute and {energy['main_mem']} pJ "
                "for loading data"
            )
            print(f"performance: {performance}")

        return real_time_ns, performance, energy

    def compute_scaled_dot_product_ns(
        self,
        key_shape: torch.Size,  # same dimensions as value_shape
        output_shape: torch.Size,
        *,
        use_kv_cache: bool = True,
        summarization: bool = False,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for scaled dot product attention."""
        batch_size = key_shape[-4] if (len(key_shape) > 3) else 1
        n_heads = key_shape[-3] if (len(key_shape) > 2) else 1
        n_rows = key_shape[-2] if (len(key_shape) > 1) else 1  # already concatenated!
        n_columns = key_shape[-1]

        q_rows = output_shape[-2] if (len(output_shape) > 2) else 1  # 1 when using kv cache in GEN.

        compute_time_ns = 0
        load_k_time = 0
        load_v_time = 0
        performance = {}
        energy = {}

        # Load KV cache if GENeration and kv cache is enabled
        # Only K is required for next step
        if not summarization and use_kv_cache:
            n_rows = self.sliding_window or n_rows
            n_heads = self.num_key_value_heads or n_heads

            k_cache = torch.Size([batch_size, n_heads, n_rows, n_columns])
            load_k_time, load_k_perf, load_k_energy = self.load_data(k_cache)
            performance = add_dictionaries(performance, load_k_perf)
            energy = add_dictionaries(energy, load_k_energy)

        # Q x Kt
        # (q_rows, embedding) x (embedding, kv_cache_length) = (q_rows, kv_cache_length)
        query_shape = torch.Size([batch_size, n_heads, q_rows, n_columns])
        kt_shape = torch.Size([batch_size, n_heads, n_columns, n_rows])
        # TODO: fix this calculation
        step_time, step_perf, step_energy = self.compute_ns(
            query_shape,
            torch.nn.Linear(in_features=n_columns, out_features=n_rows),
            kt_shape,
            load_input=False,
            load_weight=False,
        )
        compute_time_ns += max(load_k_time, step_time)  # overlap loading K with Q x Kt computation
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        # Load KV cache if GENeration and kv cache is enabled
        # Only V is required for next step
        if not summarization and use_kv_cache:
            v_cache = torch.Size([batch_size, n_heads, n_rows, n_columns])
            load_v_time, load_v_perf, load_v_energy = self.load_data(v_cache)
            performance = add_dictionaries(performance, load_v_perf)
            energy = add_dictionaries(energy, load_v_energy)

        # QxKT x V
        # (q_rows, kv_cache_length) x (kv_cache_length, embedding) = (q_rows, embedding)
        qxkt_shape = torch.Size([batch_size, n_heads, q_rows, n_rows])
        # TODO: fix this calculation
        step_time, step_perf, step_energy = self.compute_ns(
            query_shape,
            torch.nn.Linear(in_features=n_rows, out_features=n_columns),
            key_shape,
            load_input=False,
            load_weight=False,
        )
        compute_time_ns += max(
            load_v_time, step_time
        )  # overlap loading V with QxKt x V computation
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        if self.verbose:
            print(
                f"Computing scaled_dot_product: {query_shape} x {kt_shape} x {key_shape} "
                f"in {compute_time_ns} with {energy}"
            )

        return compute_time_ns, performance, energy

    def compute_matmul_ns(
        self,
        context: str,
        shape_a: torch.Size,
        shape_b: torch.Size,
        *,
        summarization: bool = False,
        sum_size: int = 0,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for matmul."""
        batch_size = shape_a[-4] if (len(shape_a) > 3) else 1
        n_heads = shape_a[-3] if (len(shape_a) > 2) else 1
        n_rows = shape_a[-2] if (len(shape_a) > 1) else 1
        n_columns = shape_a[-1]

        compute_time_ns = 0
        performance = {}
        energy = {}

        if context == "attn_weights" and not summarization:
            kv_cache = torch.Size([batch_size, n_heads, sum_size, n_columns * 2])
            step_time, step_perf, step_energy = self.load_data(kv_cache)
            compute_time_ns += step_time
            performance = add_dictionaries(performance, step_perf)
            energy = add_dictionaries(energy, step_energy)

        step_time, step_perf, step_energy = self.compute_ns(
            shape_a,
            torch.nn.Linear(in_features=n_rows, out_features=n_columns),
            shape_b,
            load_input=False,
            load_weight=False,
        )

        compute_time_ns += step_time
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        if self.verbose:
            print(f"Computing matmul: {shape_a} x {shape_b} in {compute_time_ns} with {energy}")

        return compute_time_ns, performance, energy

    def compute_activation_ns(
        self, data_shape: torch.Size, activation: str = "SiLU"
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for activation function."""
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = data_shape[-1]

        activation_ns_per_element = 0
        if activation == "SiLU":
            activation_ns_per_element = self.SiLU_ns_per_element

        tflops = batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        compute_time_ns = batch_size * n_heads * (n_rows * (activation_ns_per_element * n_columns))

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(
                f"Computing Activation: {activation} : {data_shape} in {compute_time_ns} "
                f"with {energy}"
            )

        return compute_time_ns, performance, energy

    def compute_rmsnorm_ns(
        self, data_shape: torch.Size, dimension: int
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for RMSNorm function."""
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = dimension

        tflops = batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        compute_time_ns = batch_size * n_heads * n_rows * (self.RMSNorm_ns_per_element * n_columns)

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(f"Computing RMSNorm: {data_shape} in {compute_time_ns} with {energy}")

        return compute_time_ns, performance, energy

    def compute_softmax_ns(
        self, data_shape: torch.Size
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Compute metrics for softmax function."""
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = data_shape[-1]

        tflops = batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        compute_time_ns = batch_size * n_heads * n_rows * (self.softmax_ns_per_element * n_columns)

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(f"Computing softmax: {data_shape} in {compute_time_ns} with {energy}")

        return compute_time_ns, performance, energy
