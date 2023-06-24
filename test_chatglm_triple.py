import os
os.system("pip install -r requirements_chatglm.txt")
#os.system("pip uninstall nvidia_cublas_cu11")
os.system("pip install torch==1.11.0")
os.system("pip install sentencepiece")

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/models/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/models/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "Steve Jobs founded", history=history)
print(response)
response, history = model.chat(tokenizer, "Is this true: gymnosperm genus has instance genus pseudolarix?", history=history)
print(response)
response, history = model.chat(tokenizer, "Is this true: gymnosperm genus has instance genus prociphilus?", history=history)
print(response)

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
    for line in lines:
        tmp = line.strip().split("\t")
        
        prompt = "Is this true: " + ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]] + "?"
        
        response, _ = model.chat(tokenizer, prompt, history=[])
        
        triple_txt = ent2txt[tmp[0]] + " " + rel2txt[tmp[1]] + " " + ent2txt[tmp[2]]
        print(prompt)
        print(response)
        lines_to_write.append(triple_txt +"\t" + response +"\t"+ tmp[3] + "\n")

with open("data/FB13/test_glm6b.tsv", "w") as f:
    f.writelines(lines_to_write)