import numpy as np

idx = np.random.choice(5000, 100)

ent2txt = {}
with open("data/YAGO3-10/entity2text.txt", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent2txt[tmp[0]] = tmp[1]

rel2txt = {}
with open("data/YAGO3-10/relation2text.txt", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

tail_lines_to_write = []
relation_lines_to_write = []
with open("data/YAGO3-10/test.tsv", "r", encoding = "utf-8") as f:
    lines = f.readlines()
    for i in idx:
        pos_line = lines[i]
        tmp = pos_line.strip().split("\t")
        
        prompt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]]
    
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        print(prompt + "\t" + ent2txt[tmp[2]])
        tail_lines_to_write.append(prompt + "\t" + ent2txt[tmp[2]] + "\n")

        prompt = "What is the relationship between " + ent2txt[tmp[0]] + " and " + ent2txt[tmp[2]] + "?" 
        print(prompt)
        options = "|".join([rel2txt[key] for key in rel2txt])
        easy_prompt = prompt + " Please choose your answer from: " + options + "."
        relation_lines_to_write.append(easy_prompt + "\t" + rel2txt[tmp[1]] + "\n")



with open("data/YAGO3-10/YAGO3_tail_test_human.tsv", "w", encoding = "utf-8") as f:
    f.writelines(tail_lines_to_write)

with open("data/YAGO3-10/YAGO3_relation_test_human.tsv", "w", encoding = "utf-8") as f:
    f.writelines(relation_lines_to_write)