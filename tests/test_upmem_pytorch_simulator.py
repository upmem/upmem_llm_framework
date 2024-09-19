"""Tests for `upmem_llm_framework` package."""

import warnings

import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typer.testing import CliRunner

import upmem_llm_framework as upmem_layers

runner = CliRunner()
app = typer.Typer(callback=upmem_layers.initialize_profiling_options)

gen_length = 64

layer_mapping = {
    "LlamaRMSNorm": "PIM-AI-1chip,t",
    "q_proj": "PIM-AI-4chip-duplicated,t",
    "k_proj": "PIM-AI-4chip",
    "rotatory_emb": "PIM-AI-4chip",
    "v_proj": "PIM-AI-4chip",
    "o_proj": "PIM-AI-4chip,t",
    "output_layernorm": "PIM-AI-1chip,t",
    "gate_proj": "PIM-AI-4chip,t",
    "up_proj": "PIM-AI-4chip,t",
    "down_proj": "PIM-AI-4chip,t",
    "norm": "PIM-AI-1chip,t",
    "lm_head": "PIM-AI-4chip,t",
}
layer_attn_ctxt = "q_proj"

ignored_warning = (
    "`pad_token_id` should be positive but got -1. This will cause errors when batch "
    "generating, if there is padding. Please set `pad_token_id` explicitly by "
    "`model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation, "
    "and ensure your `input_ids` input does not have negative values."
)


@app.command("profile")
def run_tiny_llama_model_with_profiler():
    """Run the tiny LLaMA model with the profiler."""

    # Initialize the profiler
    upmem_layers.profiler_init()
    # Load the tiny LLaMA model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ignored_warning)
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Prepare input data
    input_text = "Hello, world!"
    input_token = tokenizer.encode(
        input_text, return_tensors="pt", return_token_type_ids=False
    )
    input_ids = {
        "input_ids": input_token,
        "attention_mask": input_token.new_ones(input_token.shape),
    }

    # Run the profiler
    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    outputs = model.generate(
        **input_ids,
        do_sample=True,
        temperature=0.9,
        min_length=gen_length,
        max_length=gen_length,
    )
    upmem_layers.profiler_end()

    # Assert the profiler results
    assert outputs is not None
    assert outputs.shape == (1, gen_length)


def test_tiny_llama_model_with_profiler():
    """Test the tiny LLaMA model with the profiler."""
    result = runner.invoke(
        app,
        [
            "--simulation",
            "--report-layers",
            "--report-functions",
            "--print-log",
            "--print-log-summary",
            "--extra-archs=tests/extra_arch.yaml",
            "profile",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert "##### UPMEM PROFILER OUTPUT #####" in result.stdout
    assert "##### Generation Execution summary #####" in result.stdout
    assert "##### All (SUM and GEN) Execution summary #####" in result.stdout
