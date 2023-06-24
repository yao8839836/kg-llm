
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

#lines_to_write = []
triple_txt_list = []
labels = []
with open("data/WN11/test.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
        
        # response, _ = model.chat(tokenizer, prompt, history=[])
        response = ""
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        print(prompt)
        #print(response)
        #lines_to_write.append(triple_txt +"\t" + response + "\t" + tmp[3] + "\n")
        triple_txt_list.append(triple_txt)
        labels.append(tmp[3])

all_count = 0
correct_count = 0

lines_to_write = []
with open("data/WN11/test_glm6b.tsv", "r", encoding="utf-8") as f:
    all_text = f.read()

    for (i, txt) in enumerate(triple_txt_list):
        txt_idx = all_text.find(txt)
        label = labels[i]
        label_idx = all_text.find("\t" + label + "\n")
        response = all_text[txt_idx + len(txt) + 1: label_idx]

        all_text = all_text[label_idx + len("\t" + label + "\n"):]

        end_idx  = response.find(".")
        res = response[:end_idx + 1]

        print(txt, label, res)
        all_count += 1
        if label == "1" and res.find("Yes") != -1:
            correct_count += 1
        if label == "-1" and (res.find("not") != -1 or res.find("No") != -1 or res.find("n't") != -1):
            correct_count += 1
print(all_count, correct_count, 1.0 * correct_count /all_count)

