# import time
import transformers
import typer
from typing_extensions import Annotated

import upmem_llm_framework as upmem_layers

app = typer.Typer(callback=upmem_layers.initialize_profiling_options)


@app.command()
def profile(
    hf_token: Annotated[
        str, typer.Argument(envvar="hf_token", help="Hugging Face API token")
    ]
):
    upmem_layers.profiler_init()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=hf_token
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=hf_token
    )

    layer_mapping = {
        "input_layernorm": "PIM-AI-1chip",
        "q_proj": "PIM-AI-1chip",
        "k_proj": "PIM-AI-1chip",
        "rotary_emb": "PIM-AI-1chip",
        "v_proj": "PIM-AI-1chip",
        "o_proj": "PIM-AI-1chip",
        "output_layernorm": "PIM-AI-1chip",
        "gate_proj": "PIM-AI-1chip",
        "up_proj": "PIM-AI-1chip",
        "down_proj": "PIM-AI-1chip",
        "norm": "PIM-AI-1chip",
        "lm_head": "PIM-AI-1chip",
    }

    prompt = "How to prepare coffee?"

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)

    print(inputs.data["input_ids"][0].shape)
    model.eval()  # Put model in evaluation / inference mode

    # print (model)

    upmem_layers.profiler_start(layer_mapping)
    # In case we want to time the original execution (comment out profiler_start)
    # start = time.time_ns()
    gen_tokens = model.generate(
        inputs.input_ids, do_sample=True, temperature=0.9, min_length=64, max_length=64
    )
    # print ( (time.time_ns() - start)/1e6)
    upmem_layers.profiler_end()

    gen_text = tokenizer.batch_decode(
        gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(gen_text)


if __name__ == "__main__":
    app()
