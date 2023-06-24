import json
import random

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

ent2txt = {}

with open("data/WN11/entity2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent2txt[tmp[0]] = tmp[1]
rel2txt = {}

with open("data/WN11/relation2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

ent_list = []
for ent in ent2txt:
    ent_list.append(ent)

lines_to_write_glm = []
lines_to_write_llama_lora = []
with open("data/WN11/train.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
    
        tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ "Yes, this is true."+ "\"}"

        if is_json(tmp_str):
            lines_to_write_glm.append(tmp_str + "\n")
        
        tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ "Yes, this is true."+ "\"\n}"
        lines_to_write_llama_lora.append(tmp_str)

        
        rnd = random.random()

        if rnd <= 0.5:
            # corrupting head
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[0])
            tmp_ent_list = list(tmp_ent_list)
            tmp_head = random.choice(tmp_ent_list)
            prompt = "Is this true: " + ent2txt[tmp_head] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
            tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ "No, this is not true."+ "\"}"
            if is_json(tmp_str):
                lines_to_write_glm.append(tmp_str + "\n")
            
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n \"output\": \""+ "No, this is not true."+ "\"\n}"
            lines_to_write_llama_lora.append(tmp_str)

        else:
            # corrupting tail
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[2])
            tmp_ent_list = list(tmp_ent_list)
            tmp_tail = random.choice(tmp_ent_list)
            prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp_tail] + "?"
            tmp_str = "{\"prompt\": \"" + prompt + "\", \"response\": \""+ "No, this is not true."+ "\"}" 
            if is_json(tmp_str):
                lines_to_write_glm.append(tmp_str + "\n")
            
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n \"output\": \""+ "No, this is not true."+ "\"\n}"
            lines_to_write_llama_lora.append(tmp_str)
        
        print(len(lines_to_write_glm))

with open("data/WN11/train_instructions_glm.json", "w") as f:
    f.writelines(lines_to_write_glm)

with open("data/WN11/train_instructions_llama.json", "w") as f:
    tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) +"]"
    f.write(tmp_str)