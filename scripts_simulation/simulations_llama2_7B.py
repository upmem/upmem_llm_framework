# import time
from typing import Optional

import torch
import transformers
import typer
from typing_extensions import Annotated

import upmem_llm_framework as upmem_layers

app = typer.Typer(callback=upmem_layers.initialize_profiling_options)


@app.command()
def profile(
    hf_token: Annotated[
        str, typer.Argument(envvar="hf_token", help="Hugging Face API token")
    ],
    device: Annotated[str, typer.Option(help="Device to simulate for")] = "mixed",
    in_tokens: Annotated[int, typer.Option(help="Number of input tokens")] = 64,
    out_tokens: Annotated[int, typer.Option(help="Number of output tokens")] = 128,
    bs: Annotated[int, typer.Option(help="Batch size")] = 1,
):
    upmem_layers.profiler_init()

    print("Simulating with device...", device)
    print("in:", in_tokens, "out:", out_tokens)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=hf_token
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=hf_token
    )
    layer_mapping = (
        {
            "LlamaRMSNorm": "PIM-AI-1chip,t",
            "q_proj": "PIM-AI-4chip,t",
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
        if device == "mixed"
        else {
            "LlamaRMSNorm": device,
            "q_proj": device,
            "k_proj": device,
            "rotatory_emb": device,
            "v_proj": device,
            "o_proj": device,
            "output_layernorm": device,
            "gate_proj": device,
            "up_proj": device,
            "down_proj": device,
            "norm": device,
            "lm_head": device,
        }
    )
    layer_attn_ctxt = "q_proj"

    print("Batch 1")
    prompt = "placeholder"
    prompt_batch = [prompt] * bs
    input_ids = tokenizer(
        prompt_batch, return_tensors="pt", return_token_type_ids=False
    )
    input_ids["input_ids"] = torch.randint(100, [bs, in_tokens])
    input_ids["attention_mask"] = torch.ones([bs, in_tokens], dtype=torch.int)
    print(input_ids.data["input_ids"][0].shape)

    model.eval()
    print(model)
    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    # start = time.time_ns()
    gen_tokens = model.generate(
        **input_ids,
        do_sample=True,
        temperature=0.9,
        min_length=out_tokens,
        max_length=out_tokens,
    )
    # print ( (time.time_ns() - start)/1e6)
    upmem_layers.profiler_end()

    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)

    raise typer.Exit()

    print("Batch 10")
    prompt_batch = [prompt] * 10
    input_ids = tokenizer(prompt_batch, return_tensors="pt")
    print(input_ids.data["input_ids"][0].shape)

    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    gen_tokens = model.generate(
        **input_ids, do_sample=True, temperature=0.9, min_length=128, max_length=128
    )
    upmem_layers.profiler_end()

    print("Batch 30")
    prompt_batch = [prompt] * 30
    input_ids = tokenizer(prompt_batch, return_tensors="pt")
    print(input_ids.data["input_ids"][0].shape)

    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    gen_tokens = model.generate(
        **input_ids, do_sample=True, temperature=0.9, min_length=128, max_length=128
    )
    upmem_layers.profiler_end()

    print("Batch 40")
    prompt_batch = [prompt] * 40
    input_ids = tokenizer(prompt_batch, return_tensors="pt")
    print(input_ids.data["input_ids"][0].shape)

    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    gen_tokens = model.generate(
        **input_ids, do_sample=True, temperature=0.9, min_length=128, max_length=128
    )
    upmem_layers.profiler_end()

    print("Batch 200")
    prompt_batch = [prompt] * 200
    input_ids = tokenizer(prompt_batch, return_tensors="pt")
    print(input_ids.data["input_ids"][0].shape)

    upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt=layer_attn_ctxt)
    gen_tokens = model.generate(
        **input_ids, do_sample=True, temperature=0.9, min_length=128, max_length=128
    )
    upmem_layers.profiler_end()

    # Batching (from https://lukesalamone.github.io/posts/what-are-attention-masks/)
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # sentences = ["It will rain in the",
    #             "I want to eat a big bowl of",
    #             "My dog is"]
    # inputs = tokenizer(sentences, return_tensors="pt", padding=True)

    # gen_text = tokenizer.batch_decode(gen_tokens)[0]
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)

    ## torch profiler snippet
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("forward"):
    #        gen_text = tokenizer.batch_decode(gen_tokens)[0]
    #        #model(inputs)
    #
    #
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    #
    # print ("----- Group by input shape")
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    #
    # prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    app()
