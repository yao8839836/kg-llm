prompts = []
labels = []
with open("data/WN18RR/test_instructions_llama_merge.csv", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        tmp = line.strip().split("\t")
        labels.append(tmp[1])
        prompts.append(tmp[0])

correct_count = 0
with open("data/WN18RR/pred_instructions_llama.csv", "r") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.rfind(",")
        res = line[start_idx + 1:]
        
        label = labels[i]
        #print(res, label)
        if res.find(label) != -1 or label.find(res) != -1:
            correct_count += 1


print("LLaMA-13B hit@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN18RR/pred_instructions_llama13b.csv", "r") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.rfind(",")
        res = line[start_idx + 1:]
        
        label = labels[i]
        #print(res, label)
        if res.find(label) != -1 or label.find(res) != -1:
            correct_count += 1


print("LLaMA-13B hit@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN18RR/pred_instructions_llama_raw.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
        
        if res.find(label) != -1 or label.find(res) != -1:
            correct_count += 1
            #print(res, label)


print("LLaMA raw hit@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN18RR/pred_instructions_llama_raw13B.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
        
        if res.find(label) != -1 or label.find(res) != -1:
            correct_count += 1
            #print(res, label)

print("LLaMA-13B raw hit@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN18RR/generated_predictions.txt", "r", encoding= "utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.rfind(":")
        res = line[start_idx + len(": \""): -2]
        
        label = labels[i]
        #print(res, label)
        if res.find(label) != -1 or label.find(res) != -1:
            correct_count += 1
            #print(res, label, line)

print("GLM hit@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))