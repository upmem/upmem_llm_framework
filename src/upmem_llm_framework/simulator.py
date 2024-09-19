#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements multiple entry points called by the profiler to simulate the underlying
# hardware.


import torch

from upmem_llm_framework.base_architecture import BaseArchitecture
from upmem_llm_framework.sim_architectures import get_spec
from upmem_llm_framework.utils import add_dictionaries


class Simulator:

    def __init__(
        self,
        data_type_bytes=2.0,
        sliding_window=-1,
        num_key_value_heads=-1,
        verbose=False,
    ):
        self.data_type_bytes = data_type_bytes
        self.layer_mapping = {}
        self.layer_attn_ctxt = ""
        self.use_kv_cache = True
        self.sum = True
        self.sum_size = 0
        self.batch_size = 0
        self.sliding_window = sliding_window
        self.num_key_value_heads = num_key_value_heads
        self.verbose = verbose

        self.moe_already_sent = {}
        self.moe_end = ""
        self.experts_per_token = 2  # from Mixtral 8x7B

        self.current_device, _, _ = self.name_to_device("HOST")

    def start_gen(self):
        self.sum = False

    def map_layers(self, mapping, layer_attn_ctxt="", moe_end="", experts_per_token=2):
        self.layer_mapping = mapping
        self.layer_attn_ctxt = layer_attn_ctxt
        if self.verbose:
            print(self.layer_mapping)
            print(self.layer_attn_ctxt)
        self.moe_end = moe_end
        self.experts_per_token = experts_per_token

    def name_to_device(self, full_name: str):

        device_name = full_name.split(",")[0].replace("-", "_")
        flags = full_name.split(",")[1] if len(full_name.split(",")) > 1 else ""

        do_transfer = "t" in flags
        moe = "m" in flags

        new_device = BaseArchitecture(
            data_type_bytes=self.data_type_bytes,
            sliding_window=self.sliding_window,
            num_key_value_heads=self.num_key_value_heads,
            verbose=self.verbose,
        )

        if device_name == "HOST":
            new_device.name = "HOST"
            return new_device, do_transfer, moe

        spec = get_spec(device_name)
        new_device.load_spec(device_name, spec)

        # new_device.adjust_for_quantization()

        return new_device, do_transfer, moe

    def simulate_attn(self, input_shape):
        batch_size = input_shape[0] if (len(input_shape) > 2) else 1
        n_rows = input_shape[1] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        # input (n_rows, n_columns) x Wqkv (n_columns, Wqkv),
        # where Wqkv is n_columns*3 (all Wq, Wk, Wv together)
        # Qall_heads = (n_rows, n_columns), each head computes n_columns/n_heads
        # Kall_heads = (n_rows, n_columns)
        # Vall_heads = (n_rows, n_columns)

        # TODO: add transpose time, concat time
        compute_time_ns = 0
        performance = {}
        energy_compute = {}

        kt = torch.Size([n_columns, n_rows])
        qkt = torch.Size([batch_size, n_rows, n_rows])
        v = torch.Size([n_rows, n_columns])

        if self.use_kv_cache:
            if self.sum:
                self.sum_size = n_rows  # Just to keep it updated
            else:
                # load KV cache
                kv_cache = torch.Size([batch_size, self.sum_size, n_columns * 2])
                step_time, step_perf, step_energy = self.current_device.load_data(
                    kv_cache
                )
                compute_time_ns += step_time
                performance = add_dictionaries(performance, step_perf)
                energy_compute = add_dictionaries(energy_compute, step_energy)

                # concat K + 1
                kt = torch.Size([n_columns, self.sum_size + 1])
                # concat V + 1
                v = torch.Size([self.sum_size + 1, n_columns])
                qkt = torch.Size([batch_size, n_rows, self.sum_size + 1])

        # Q x Kt
        if self.verbose:
            print("Computing Q x Kt")
            step_time, step_perf, step_energy = self.current_device.compute_ns(
                input_shape,
                torch.nn.Linear(in_features=n_columns, out_features=n_rows),
                kt,
                load_input=False,
            )
        compute_time_ns += step_time
        performance = add_dictionaries(performance, step_perf)
        energy_compute = add_dictionaries(energy_compute, step_energy)

        # output = V * QKt
        if self.verbose:
            print("Computing V x QKt")
        step_time, step_perf, step_energy = self.current_device.compute_ns(
            qkt,
            torch.nn.Linear(in_features=n_rows, out_features=n_rows),
            v,
            load_input=False,
        )
        compute_time_ns += step_time
        performance = add_dictionaries(performance, step_perf)
        energy_compute = add_dictionaries(energy_compute, step_energy)

        if self.verbose:
            print(
                f"Attn ({'SUM.' if self.sum else 'GEN.'}): "
                f"Q: {input_shape} Kt: {kt} QKt: {qkt} V: {v}"
            )
            print(performance)

        return compute_time_ns, performance, energy_compute

    def simulate_end(self, input_shape, generated_tokens=1):

        time_send_ans_to_host = 0
        perf_send_ans_to_host = {}
        energy_send_ans_to_host = {}
        data_send_ans_to_host = {}

        if self.current_device.name != "HOST":
            # we asume the new input is what needs to be written back to host from previous layer
            (
                time_send_ans_to_host,
                perf_send_ans_to_host,
                energy_send_ans_to_host,
                data_send_ans_to_host,
            ) = self.current_device.host_transfer(
                input_shape, direction="to_host", generated_tokens=generated_tokens
            )

        return (
            time_send_ans_to_host,
            perf_send_ans_to_host,
            energy_send_ans_to_host,
            data_send_ans_to_host,
        )

    def check_moe(self, context):
        num_seen = self.moe_already_sent.get(context, 0)

        self.moe_already_sent[context] = num_seen + 1

        return num_seen == 0

    def reset_moe(self):
        all_moe_seen = True
        for v in self.moe_already_sent.values():
            all_moe_seen = all_moe_seen and (v == self.experts_per_token)

        if all_moe_seen:
            # Reset dict for next iteration
            self.moe_already_sent = {}

        return all_moe_seen

    def check_sync_point(self, context, input_shape):
        time_send_ans_to_host = 0
        time_send_ans_from_host = 0
        perf = {}
        energy = {}
        moved_data = {}

        new_device = None

        # print(f"Try mapping {context}")
        # assume that if the layer is not mapped, it stays in the current device
        if context in self.layer_mapping:
            # print(f"Mapping {context} to {self.layer_mapping[context]}")
            new_device, gather_at_host, moe = self.name_to_device(
                self.layer_mapping[context]
            )

            # if new_device != current_device --> pay transfer
            if (
                new_device.name != self.current_device.name
                or gather_at_host
                or (moe and self.check_moe(context))
            ):

                # if HOST is not current device, transfer to HOST the output from the last layer
                # we asume the new input is what needs to be written back to host from previous
                # layer
                if self.current_device.name != "HOST":
                    time_send_ans_to_host, step_perf, step_energy, step_data = (
                        self.current_device.host_transfer(
                            input_shape, direction="to_host"
                        )
                    )
                    perf = add_dictionaries(perf, step_perf)
                    energy = add_dictionaries(energy, step_energy)
                    moved_data = add_dictionaries(moved_data, step_data)

                # change current_device = new_device
                old_device = self.current_device
                self.current_device = new_device

                # then, host writes into the new device
                if self.current_device.name != "HOST":
                    time_send_ans_from_host, step_perf, step_energy, step_data = (
                        self.current_device.host_transfer(input_shape)
                    )
                    perf = add_dictionaries(perf, step_perf)
                    energy = add_dictionaries(energy, step_energy)
                    moved_data = add_dictionaries(moved_data, step_data)

                if self.verbose:
                    print(
                        f"Changing device from {old_device.name} to {new_device.name} "
                        f"took {perf} with energy: {energy}"
                    )

        return time_send_ans_to_host, time_send_ans_from_host, perf, energy, moved_data

    def simulate_layer(self, layer, input_shape, layer_obj, weight_shape, output_shape):

        time_send_ans_to_host = 0
        time_send_ans_from_host = 0
        compute_time_ns = 0

        perf_transfer = {}
        perf_compute = {}

        energy_transfer = {}
        energy_compute = {}

        data_transfer = {}

        if self.verbose:
            print("Simulating layer:", layer.context, "n_layer:", layer.n_layer)

        (
            time_send_ans_to_host,
            time_send_ans_from_host,
            perf_transfer,
            energy_transfer,
            data_transfer,
        ) = self.check_sync_point(layer.context, input_shape)

        if layer.context == self.moe_end and self.moe_end != "":
            if self.reset_moe():
                # output_shape expected to be [tokens * batch_size, features]
                # Send all experts' output except one, which shall be accounted by the layer mapping
                transfer_shape = torch.Size(
                    [self.experts_per_token - 1, output_shape[0], output_shape[1]]
                )
                (
                    time_send_ans_to_host_moe,
                    perf_transfer_moe,
                    energy_transfer_moe,
                    data_transfer_moe,
                ) = self.current_device.host_transfer(
                    transfer_shape, direction="to_host"
                )
                if self.verbose:
                    print("Last layer of MoE sends back to HOST: ", transfer_shape)
                time_send_ans_to_host += time_send_ans_to_host_moe
                perf_transfer = add_dictionaries(perf_transfer, perf_transfer_moe)
                energy_transfer = add_dictionaries(energy_transfer, energy_transfer_moe)
                data_transfer = add_dictionaries(data_transfer, data_transfer_moe)

        # pay compute
        step_time, step_perf, step_energy = self.current_device.compute_ns(
            input_shape, layer_obj, weight_shape
        )
        compute_time_ns += step_time
        perf_compute = add_dictionaries(perf_compute, step_perf)
        energy_compute = add_dictionaries(energy_compute, step_energy)

        # If self-attention required, simulate
        # if (layer.context == self.layer_attn_ctxt):
        #     step_time, step_perf, step_energy = self.simulate_attn(input_shape, weight_shape)
        #     compute_time_ns += step_time
        #     perf_compute     = add_dictionaries(perf_compute  , step_perf  )
        #     energy_compute   = add_dictionaries(energy_compute, step_energy)

        if self.verbose:
            print("Time send ans to host (ns):", time_send_ans_to_host)
            print("Time send ans from host (ns):", time_send_ans_from_host)
            print("Compute time (ns):", compute_time_ns)
            print("Energy send ans to/from host (pJ):", energy_transfer)
            print("Energy Compute (pJ):", energy_compute)

        # we can pipeline TODO: calculate this
        # max (time_send_ans_from_host, compute_time)

        # return simulated stats
        total_time = time_send_ans_to_host + time_send_ans_from_host + compute_time_ns

        total_perf = add_dictionaries(perf_transfer, perf_compute)

        total_energy = add_dictionaries(energy_transfer, energy_compute)

        return total_time, total_perf, total_energy, data_transfer

    def simulate_function(self, function, context, input_shape, output_shape):

        function_name = (
            function.__name__ if hasattr(function, "__name__") else function.name
        )

        if self.verbose:
            print(
                f"Simulating function: {function_name}, "
                f"context: {context}, "
                f"input shape: {input_shape}, "
                f"output shape: {output_shape}"
            )

        (
            time_send_ans_to_host,
            time_send_ans_from_host,
            perf_transfer,
            energy_transfer,
            data_transfer,
        ) = self.check_sync_point(function_name, input_shape)

        compute_time_ns, perf_compute, energy_compute = self._compute_function_metrics(
            function, context, input_shape, output_shape
        )

        if self.verbose:
            print("Time send ans to host (ns):", time_send_ans_to_host)
            print("Time send ans from host (ns):", time_send_ans_from_host)
            print("Compute time (ns):", compute_time_ns)
            print("Energy send ans to/from host (pJ):", energy_transfer)
            print("Energy Compute (pJ):", energy_compute)

        # return simulated stats
        total_time = time_send_ans_to_host + time_send_ans_from_host + compute_time_ns

        total_perf = add_dictionaries(perf_transfer, perf_compute)
        total_energy = add_dictionaries(energy_transfer, energy_compute)

        return total_time, total_perf, total_energy, data_transfer

    def _compute_function_metrics(self, function, context, input_shape, output_shape):
        if hasattr(function, "__name__") and function.__name__.endswith("softmax"):
            return self.current_device.compute_softmax_ns(input_shape)
        if hasattr(function, "name") and function.name.endswith("LlamaRMSNorm"):
            return self.current_device.compute_RMSNorm_ns(input_shape, output_shape)
        if hasattr(function, "name") and (
            function.name.endswith("SiLU") or function.name.endswith("SiLUActivation")
        ):
            return self.current_device.compute_activation_ns(
                input_shape, activation="SiLU"
            )
        if hasattr(function, "__name__") and function.__name__.endswith("matmul"):
            return self.current_device.compute_matmul_ns(
                context,
                input_shape,
                output_shape,
                summarization=self.sum,
                sum_size=self.sum_size,
            )
        if hasattr(function, "__name__") and function.__name__.endswith(
            "scaled_dot_product_attention"
        ):
            return self.current_device.compute_scaled_dot_product_ns(
                context,
                input_shape,
                output_shape,
                summarization=self.sum,
                sum_size=self.sum_size,
            )
        raise ValueError(
            "Unsupported function: "
            f"{function.__name__ if hasattr(function, '__name__') else function.name}, "
            f"type: {type(function)}, "
            f"string: {function}"
        )
