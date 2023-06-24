import numpy as np

idx = np.random.choice(23733, 50)

ent2txt = {}
with open("data/FB13/entity2text_capital.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent2txt[tmp[0]] = tmp[1]

rel2txt = {}
with open("data/FB13/relation2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

lines_to_write = []
with open("data/FB13/test.tsv", "r") as f:
    lines = f.readlines()
    for i in idx:
        pos_line = lines[2*i]
        tmp = pos_line.strip().split("\t")
        
        prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
        
        response = ""
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        print(tmp[3] + "\t" + prompt)
        lines_to_write.append(tmp[3] + "\t" + prompt + "\n")

        neg_line = lines[2*i + 1]
        tmp = neg_line.strip().split("\t")
        
        prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
        
        response = ""
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        print(tmp[3] + "\t" + prompt)
        lines_to_write.append(tmp[3] + "\t" + prompt + "\n")


with open("data/FB13/FB13_test_human.tsv", "w") as f:
    f.writelines(lines_to_write)