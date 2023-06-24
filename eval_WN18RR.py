import os


ent2txt = {}

with open("data/WN18RR/entity2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent2txt[tmp[0]] = tmp[1]
rel2txt = {}

with open("data/WN18RR/relation2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

lines_to_write = []
triple_txt_list = []
labels_t = []
with open("data/WN18RR/test.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]]
        
        #response, _ = model.chat(tokenizer, prompt, history=[])
        #print(response)
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        #print(triple_txt)
        labels_t.append(ent2txt[tmp[2]])
        #lines_to_write.append(triple_txt +"\t" + response + "\n")
        triple_txt_list.append(triple_txt)


all_count = 0
correct_count = 0

lines_to_write = []
with open("data/WN18RR/test_glm6b_t.tsv", "r", encoding="utf-8") as f:
    all_text = f.read()

    for (i, txt) in enumerate(triple_txt_list):
        txt_idx = all_text.find(txt)
        label = labels_t[i]
        label_idx = all_text.find("\t" + label + "\n")
        response = all_text[txt_idx + len(txt) + 1: label_idx]

        all_text = all_text[label_idx + len("\t" + label + "\n"):]

        end_idx  = response.find(".")
        res = response[:end_idx + 1]

        print("Text: ", txt)
        
        comma_idx = label.find(",")
        label = label[:comma_idx]
        print("Label: ", label)
        print("Response: ", res)
        all_count += 1
        if res.find(label) != -1:
            correct_count += 1

print(all_count, correct_count, 1.0 * correct_count /all_count)
