import json
import numpy as np

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

ent2txt = {}

with open("data/WN18RR/entity2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        comma_idx = tmp[1].find(",")
        ent2txt[tmp[0]] = tmp[1][:comma_idx]
rel2txt = {}

with open("data/WN18RR/relation2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

ent_list = []
for ent in ent2txt:
    ent_list.append(ent)

tail_lines_to_write_glm = []
tail_lines_to_write_llama_lora = []

head_lines_to_write_glm = []
head_lines_to_write_llama_lora = []
with open("data/WN18RR/train.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]]
    
        tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ ent2txt[tmp[2]] + "\"}"

        if is_json(tmp_str):
            tail_lines_to_write_glm.append(tmp_str + "\n")
        
        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ ent2txt[tmp[2]] + "\"\n}"
        tail_lines_to_write_llama_lora.append(tmp_str)

        print(len(tail_lines_to_write_glm))

        prompt = "What/Who/When/Where/Why" + " " + rel2txt[tmp[1]] + " " +  ent2txt[tmp[2]] + "?"
        tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ ent2txt[tmp[0]] + "\"}"
        if is_json(tmp_str):
            head_lines_to_write_glm.append(tmp_str + "\n")

        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ ent2txt[tmp[0]] + "\"\n}"
        head_lines_to_write_llama_lora.append(tmp_str)

# GLM
with open("data/WN18RR/train_instructions_glm_tail.json", "w") as f:
    f.writelines(tail_lines_to_write_glm)

with open("data/WN18RR/train_instructions_glm_head.json", "w") as f:
    f.writelines(head_lines_to_write_glm)

lines_to_write_glm_merge = tail_lines_to_write_glm + head_lines_to_write_glm
np.random.seed(42)
np.random.shuffle(lines_to_write_glm_merge)

with open("data/WN18RR/train_instructions_glm_merge.json", "w") as f:
    f.writelines(lines_to_write_glm_merge)

# LLaMA
with open("data/WN18RR/train_instructions_llama_tail.json", "w") as f:
    tmp_str = "[\n" + ",\n".join(tail_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)

with open("data/WN18RR/train_instructions_llama_head.json", "w") as f:
    tmp_str = "[\n" + ",\n".join(head_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)

lines_to_write_llama_lora = tail_lines_to_write_llama_lora + head_lines_to_write_llama_lora
np.random.seed(42)
np.random.shuffle(lines_to_write_llama_lora)

with open("data/WN18RR/train_instructions_llama_merge.json", "w") as f:
    tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) +"]"
    f.write(tmp_str)