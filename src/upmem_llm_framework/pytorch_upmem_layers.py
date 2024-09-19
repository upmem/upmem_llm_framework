#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file wraps PyTorch classes and functions into new UPM classes and functions able to
# track the start, inputs, end and outputs of the corresponding function.
# Currently, forward from multiple modules and other minor functions
# (normalizations, softmax, activations, etc.) are tracked and profiled.

from inspect import getframeinfo, stack
from typing import Tuple

import torch
import transformers

from upmem_llm_framework.options import options
from upmem_llm_framework.profiler import UPM_Profiler

profiler: UPM_Profiler = None
profiling = 0


def get_context():
    # https://stackoverflow.com/questions/24438976/debugging-get-filename-and-line-number-from-which-a-function-is-called
    return getframeinfo(stack()[2][0]).code_context[0].split()[0].replace("self.", "")


class UPM_Module(torch.nn.Module):

    def forward(self, x):
        x = super().forward(x)
        return x


class UPM_Linear(torch.nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_NonDynamicallyQuantizableLinear(
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())
        print("HERE")

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_LayerNorm(torch.nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_Embedding(torch.nn.Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_LlamaRotaryEmbedding(
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        x = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)
        return x


class UPM_LlamaRMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, hidden_states):
        context = get_context()
        profiler.forward_start(hidden_states.shape)
        hidden_states = super().forward(hidden_states)
        profiler.forward_end(hidden_states.shape, context, layer_obj=self)
        return hidden_states


class UPM_SiLUActivation(torch.nn.SiLU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_NewGELUActivation(transformers.activations.NewGELUActivation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


# Not used in inference
class UPM_Dropout(torch.nn.Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())


class UPM_Conv1d(torch.nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())


class UPM_Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_Conv1D(transformers.pytorch_utils.Conv1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context)
        return x


class UPM_Softmax(torch.nn.Softmax):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context)
        return x


class UPM_Tensor(torch.Tensor):

    def transpose(self, dim0, dim1):
        print("MyTranpose with input:", self, "dim0", dim0, "dim1", dim1)
        super().transpose(dim0, dim1)


__pytorch_nn_functional_softmax = torch.nn.functional.softmax


# TODO: change logic here to not use stringly types
def UPM_Softmax_functional(input, dim=None, dtype=None):
    context = get_context()
    profiler.forward_func_start("softmax", context, input.shape)
    x = __pytorch_nn_functional_softmax(input, dim=dim, dtype=dtype)
    profiler.forward_func_end(__pytorch_nn_functional_softmax, context, x.shape)
    return x


__pytorch_matmul = torch.matmul


# TODO: here too
def UPM_Matmul(input, other, *, out=None):
    context = get_context()
    profiler.forward_func_start("matmul", context, input.shape)
    x = __pytorch_matmul(input, other, out=out)
    profiler.forward_func_end(__pytorch_matmul, context, x.shape)
    return x


__pytorch_scaled_dot_product_attention = (
    torch.nn.functional.scaled_dot_product_attention
)


# TODO: here too
def UPM_scaled_dot_product_attention(query, key, value, **kwargs):
    context = get_context()
    # profiler.forward_func_start("scaled_dot_product_attention", context, [query.shape, key.shape, value.shape])
    profiler.forward_func_start("scaled_dot_product_attention", context, key.shape)
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


def UPM_Transpose(input, dim0, dim1):
    print("UPM_Transpose with input", input.shape, "dim0:", dim0, "dim1", dim1)
    x = __pytorch_transpose(input, dim0, dim1)
    return x


def profiler_init():

    print(f"Options: {options}")

    global profiling, profiler
    profiling = 1
    profiler = UPM_Profiler(options)

    # torch library
    torch.nn.Module = UPM_Module
    torch.nn.Linear = UPM_Linear
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear = (
        UPM_NonDynamicallyQuantizableLinear
    )
    torch.nn.LayerNorm = UPM_LayerNorm
    torch.nn.Embedding = UPM_Embedding
    torch.nn.Dropout = UPM_Dropout
    torch.nn.Conv1d = UPM_Conv1d
    torch.nn.Conv2d = UPM_Conv2d
    torch.nn.Softmax = UPM_Softmax
    torch.nn.functional.softmax = UPM_Softmax_functional
    torch.matmul = UPM_Matmul
    torch.transpose = UPM_Transpose
    torch.nn.functional.scaled_dot_product_attention = UPM_scaled_dot_product_attention
    # torch.Tensor = UPM_Tensor

    # transformers library
    transformers.pytorch_utils.Conv1D = UPM_Conv1D
    transformers.activations.NewGELUActivation = UPM_NewGELUActivation
    transformers.activations.ACT2FN["gelu_new"] = (
        UPM_NewGELUActivation  # classes are hardcoded in ACT2FN
    )
    transformers.activations.ACT2FN["silu"] = (
        UPM_SiLUActivation  # classes are hardcoded in ACT2FN
    )
    transformers.models.llama.modeling_llama.LlamaRMSNorm = UPM_LlamaRMSNorm
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = (
        UPM_LlamaRotaryEmbedding
    )

    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm = (
        UPM_LlamaRMSNorm  # miXtral models
    )
    transformers.models.mistral.modeling_mistral.MistralRMSNorm = (
        UPM_LlamaRMSNorm  # miStral models
    )


def profiler_start(
    layer_mapping=None,
    layer_attn_ctxt="",
    last_layer="lm_head",
    batch_size=1,
    moe_end="",
    experts_per_token=2,
):
    profiler.start(
        layer_mapping=layer_mapping,
        layer_attn_ctxt=layer_attn_ctxt,
        last_layer=last_layer,
        batch_size=batch_size,
        moe_end=moe_end,
        experts_per_token=experts_per_token,
    )


def profiler_end():
    profiler.end()
