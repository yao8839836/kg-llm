import json
import numpy as np

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

ent2txt = {}

with open("data/YAGO3-10/entity2text.txt", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        #comma_idx = tmp[1].find(",")
        ent2txt[tmp[0]] = tmp[1]
rel2txt = {}

with open("data/YAGO3-10/relation2text.txt", "r", encoding = "utf-8") as f:
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

rel_lines_to_write_glm = []
rel_lines_to_write_llama_lora = []

with open("data/YAGO3-10/train.tsv", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]]
    
        tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ ent2txt[tmp[2]] + "\"}"

        if is_json(tmp_str):
            tail_lines_to_write_glm.append(tmp_str + "\n")
        
        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ ent2txt[tmp[2]] + "\"\n}"
        tail_lines_to_write_llama_lora.append(tmp_str)

        #print(len(lines_to_write_glm))

        prompt = "What/Who/When/Where/Why" + " " + rel2txt[tmp[1]] + " " +  ent2txt[tmp[2]] + "?"
        tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ ent2txt[tmp[0]] + "\"}"
        if is_json(tmp_str):
            head_lines_to_write_glm.append(tmp_str + "\n")

        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ ent2txt[tmp[0]] + "\"\n}"
        head_lines_to_write_llama_lora.append(tmp_str)

        prompt = "What is the relationship between" + " " + ent2txt[tmp[0]] + " and " +  ent2txt[tmp[2]] + "?"
        options = "|".join([rel2txt[key] for key in rel2txt])
        easy_prompt = prompt + " Please choose your answer from: " + options + "."

        tmp_str = "{\"prompt\": \"" + easy_prompt + "\", \"response\": \""+ rel2txt[tmp[1]] + "\"}"
        if is_json(tmp_str):
            rel_lines_to_write_glm.append(tmp_str + "\n")

        tmp_str = "{\n\"instruction\": \"" + easy_prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ rel2txt[tmp[1]] + "\"\n}"
        rel_lines_to_write_llama_lora.append(tmp_str)

with open("data/YAGO3-10/train_instructions_glm_tail.json", "w", encoding = "utf-8") as f:
    f.writelines(tail_lines_to_write_glm)

with open("data/YAGO3-10/train_instructions_glm_head.json", "w", encoding = "utf-8") as f:
    f.writelines(head_lines_to_write_glm)

lines_to_write_glm_merge = tail_lines_to_write_glm + head_lines_to_write_glm
np.random.seed(42)
np.random.shuffle(lines_to_write_glm_merge)

with open("data/YAGO3-10/train_instructions_glm_merge.json", "w", encoding = "utf-8") as f:
    f.writelines(lines_to_write_glm_merge)


with open("data/YAGO3-10/train_instructions_llama_tail.json", "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(tail_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)

with open("data/YAGO3-10/train_instructions_llama_head.json", "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(head_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)

lines_to_write_llama_lora = tail_lines_to_write_llama_lora + head_lines_to_write_llama_lora
np.random.seed(42)
np.random.shuffle(lines_to_write_llama_lora)

with open("data/YAGO3-10/train_instructions_llama_merge.json", "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) +"]"
    f.write(tmp_str)

with open("data/YAGO3-10/train_instructions_glm_rel.json", "w", encoding = "utf-8") as f:
    f.writelines(rel_lines_to_write_glm)

with open("data/YAGO3-10/train_instructions_llama_rel.json", "w", encoding = "utf-8") as f:
    tmp_str = "[\n" + ",\n".join(rel_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)