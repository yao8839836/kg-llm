import sys
import torch
import argparse
import pandas as pd
from peft import PeftModel
import transformers
import gradio as gr
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
tokenizer = LlamaTokenizer.from_pretrained("models/LLaMA-HF/tokenizer/")
LOAD_8BIT = False
BASE_MODEL = "models/LLaMA-HF/llama-13b"
LORA_WEIGHTS = "models/llama-13b-lora-wn11"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        # torch_dtype=torch.float16,
        # device_map="auto",
    ).half().cuda()
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        # torch_dtype=torch.float16,
    ).half().cuda()
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""
# if not LOAD_8BIT:
    # model.half()  # seems to fix bugs for some users.
model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)
def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

if __name__ == "__main__":
    # testing code for readme
    parser = argparse.ArgumentParser()
    parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama13b.csv")
    args = parser.parse_args()
    total_input = pd.read_csv(args.finput, header=0, sep='\t')
    instruct, pred = [], []
    for index, data in total_input.iterrows():
        cur_instruct = data['prompt']
        cur_response = evaluate(cur_instruct)
        pred.append(cur_response)
        instruct.append(cur_instruct)
    
    output = pd.DataFrame({'prompt': instruct, 'generated': pred})
    output.to_csv(args.foutput, header=True, index=False)
