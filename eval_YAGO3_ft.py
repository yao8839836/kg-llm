labels = []
prompts = []
with open("data/YAGO3-10/test_instructions_llama_merge.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for line in lines[1:]:
        tmp = line.strip().split("\t")
        labels.append(tmp[1])
        prompts.append(tmp[0])

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_link.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.rfind(",")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1


print("LLaMA7B link prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_link13b.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.rfind(",")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1


print("LLaMA13B link prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_link_raw.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1
            #print(line, label)


print("LLaMA-7B raw link prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))


correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_raw13B.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1
            #print(line, label)


print("LLaMA-13B raw link prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

labels = []
prompts = []
with open("data/YAGO3-10/test_instructions_llama_rel.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for line in lines[1:]:
        tmp = line.strip().split("\t")
        labels.append(tmp[1])
        prompts.append(tmp[0])

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_rel13b.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.rfind(",")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1
            
print("LLaMA13B relation prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_rel_raw.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
       
        if res.find(label) != -1 and res.find("Please choose your answer from:") == -1:
            correct_count += 1
            #print(label, res)
            

print("LLaMA-7B raw relation prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/YAGO3-10/pred_instructions_llama_rel_raw13B.csv", "r", encoding="utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find(prompts[i])
        res = line[start_idx + len(prompts[i]):]
        
        label = labels[i]
       
        if res.find(label) != -1 and res.find("Please choose your answer from:") == -1:
            correct_count += 1
            #print(label, res)
            

print("LLaMA-13B raw relation prediction hits@1: ", correct_count, len(labels), 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/YAGO3-10/generated_predictions.txt", "r", encoding= "utf=8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.rfind(":")
        res = line[start_idx + len(": \""): -2]
    

        start_idx = line.find(":")
        end_idx = line.find(", \"predict\"")
        label = line[start_idx + len(": \""): end_idx-1]
        #print(res, label, line)
        if res.find(label) != -1:
            correct_count += 1
            #print(res, label, line)

print("GLM link hit@1: ", correct_count, len(lines), 1.0 * correct_count/len(lines))

correct_count = 0
with open("data/YAGO3-10/r_generated_predictions.txt", "r", encoding= "utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.rfind(":")
        res = line[start_idx + len(": \""): -2]
    

        start_idx = line.find(":")
        end_idx = line.find(", \"predict\"")
        label = line[start_idx + len(": \""): end_idx-1]
        #print(res, label, line)
        if res.find(label) != -1:
            correct_count += 1
            #print(res, label, line)

print("GLM relation hit@1: ", correct_count, len(lines), 1.0 * correct_count/len(lines))

correct_count = 0
with open("data/YAGO3-10/raw_r_generated_predictions.txt", "r", encoding= "utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.rfind(":")
        res = line[start_idx + len(": \""): -2]
    

        start_idx = line.find(":")
        end_idx = line.find(", \"predict\"")
        label = line[start_idx + len(": \""): end_idx-1]
        #print(res, label, line)
        if res.find(label) != -1:
            correct_count += 1
            #print(res, label, line)

print("GLM raw relation hit@1: ", correct_count, len(lines), 1.0 * correct_count/len(lines))