"""Wrap PyTorch classes and functions into new UPM classes and functions.

Those are able to track the start, inputs, end and outputs of the corresponding function.
Currently, forward from multiple modules and other minor functions
(normalizations, softmax, activations, etc.) are tracked and profiled.
"""

from inspect import getframeinfo, stack

import torch
import transformers
import transformers.activations
import transformers.models
from torch import Tensor, dtype

from upmem_llm_framework.options import options
from upmem_llm_framework.profiler import UPMProfiler

profiler: UPMProfiler = UPMProfiler(options)
profiling = 0


def _get_context() -> str:
    # https://stackoverflow.com/questions/24438976/debugging-get-filename-and-line-number-from-which-a-function-is-called
    frame_info = getframeinfo(stack()[2][0])
    if frame_info.code_context and frame_info.code_context[0]:
        return frame_info.code_context[0].split()[0].replace("self.", "")
    return ""


class UPMModule(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class UPMLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMNonDynamicallyQuantizableLinear(torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


_shape_t = int | list[int] | torch.Size


class UPMLayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMLlamaRotaryEmbedding(transformers.models.llama.modeling_llama.LlamaRotaryEmbedding):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = _get_context()
        shape = x.shape
        profiler.forward_start(shape)
        y = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)
        return y


class UPMLlamaRMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, hidden_states: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(hidden_states.shape)
        hidden_states = super().forward(hidden_states)
        profiler.forward_end(hidden_states.shape, context, layer_obj=self)
        return hidden_states


class UPMSiLUActivation(torch.nn.SiLU):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMNewGELUActivation(transformers.activations.NewGELUActivation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


# Not used in inference
class UPMDropout(torch.nn.Dropout):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())


class UPMConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())


class UPMConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMConv1D(transformers.pytorch_utils.Conv1D):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMSoftmax(torch.nn.Softmax):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        profiler.add(self, _get_context())

    def forward(self, x: Tensor) -> Tensor:
        context = _get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPMTensor(torch.Tensor):
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        print("UPMTranpose with input:", self, "dim0", dim0, "dim1", dim1)
        return super().transpose(dim0, dim1)


__pytorch_nn_functional_softmax = torch.nn.functional.softmax


# TODO: change logic here to not use stringly types
def upm_softmax_functional(
    input: Tensor, dim: int | None = None, dtype: dtype | None = None
) -> Tensor:
    context = _get_context()
    profiler.forward_func_start(input.shape)
    x = __pytorch_nn_functional_softmax(input, dim=dim, dtype=dtype)
    profiler.forward_func_end(__pytorch_nn_functional_softmax, context, x.shape)
    return x


__pytorch_matmul = torch.matmul


# TODO: here too
def upm_matmul(input: Tensor, other: Tensor, *, out: Tensor | None = None) -> Tensor:
    context = _get_context()
    profiler.forward_func_start(input.shape)
    x = __pytorch_matmul(input, other, out=out)
    profiler.forward_func_end(__pytorch_matmul, context, x.shape)
    return x


__pytorch_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


# TODO: here too
def upm_scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs) -> Tensor:
    context = _get_context()
    profiler.forward_func_start(key.shape)
    if options.sim_compute:
        q_shape = list(query.shape)
        v_shape = list(value.shape)
        q_shape[-1] = v_shape[-1]
        x = torch.zeros(q_shape)
    else:
        x = __pytorch_scaled_dot_product_attention(query, key, value, **kwargs)
    profiler.forward_func_end(__pytorch_scaled_dot_product_attention, context, x.shape)
    return x


__pytorch_transpose = torch.transpose


def upm_transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    print("upm_transpose with input", input.shape, "dim0:", dim0, "dim1", dim1)
    return __pytorch_transpose(input, dim0, dim1)


def profiler_init() -> None:
    """Initialize the profiler and wrap PyTorch classes and functions."""
    print(f"Options: {options}")

    global profiling, profiler
    profiling = 1
    profiler = UPMProfiler(options)

    # torch library
    torch.nn.Module = UPMModule
    torch.nn.Linear = UPMLinear
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear = UPMNonDynamicallyQuantizableLinear
    torch.nn.LayerNorm = UPMLayerNorm
    torch.nn.Embedding = UPMEmbedding
    torch.nn.Dropout = UPMDropout
    torch.nn.Conv1d = UPMConv1d
    torch.nn.Conv2d = UPMConv2d
    torch.nn.Softmax = UPMSoftmax
    torch.nn.functional.softmax = upm_softmax_functional
    torch.matmul = upm_matmul
    torch.transpose = upm_transpose
    torch.nn.functional.scaled_dot_product_attention = upm_scaled_dot_product_attention
    # torch.Tensor = UPM_Tensor

    # transformers library
    transformers.pytorch_utils.Conv1D = UPMConv1D
    transformers.activations.NewGELUActivation = UPMNewGELUActivation
    transformers.activations.ACT2FN["gelu_new"] = (
        UPMNewGELUActivation  # classes are hardcoded in ACT2FN
    )
    transformers.activations.ACT2FN["silu"] = UPMSiLUActivation  # classes are hardcoded in ACT2FN
    transformers.models.llama.modeling_llama.LlamaRMSNorm = UPMLlamaRMSNorm
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = UPMLlamaRotaryEmbedding

    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm = UPMLlamaRMSNorm  # miXtral models
    transformers.models.mistral.modeling_mistral.MistralRMSNorm = UPMLlamaRMSNorm  # miStral models


def profiler_start(
    layer_mapping: dict[str, str] | None = None,
    layer_attn_ctxt: str = "",
    last_layer: str = "lm_head",
    batch_size: int = 1,
    moe_end: str = "",
    experts_per_token: int = 2,
) -> None:
    """Start the profiler with the given parameters."""
    if layer_mapping is None:
        layer_mapping = {}

    profiler.start(
        layer_mapping=layer_mapping,
        layer_attn_ctxt=layer_attn_ctxt,
        last_layer=last_layer,
        batch_size=batch_size,
        moe_end=moe_end,
        experts_per_token=experts_per_token,
    )


def profiler_end() -> None:
    """End the profiler and print the report."""
    profiler.end()
