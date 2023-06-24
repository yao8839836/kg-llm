
labels = []
with open("data/FB13/test.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        labels.append(tmp[3])


correct_count = 0
with open("data/FB13/pred_instructions_llama.csv", "r") as f:
    lines = f.readlines()
    print(len(lines))
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("\"")
        res = line[start_idx + 1: -1]
        
        label = labels[i]
        #print(res, label)
        if res.find("Yes") != -1 and label == "1":
            correct_count += 1
        elif res.find("No") != -1 and label == "-1":
            correct_count += 1

print("LLaMA-7B acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/FB13/pred_instructions_llama13b.csv", "r") as f:
    lines = f.readlines()
    print(len(lines))
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("\"")
        res = line[start_idx + 1: -1]
        
        label = labels[i]
        #print(res, label)
        if res.find("Yes") != -1 and label == "1":
            correct_count += 1
        elif res.find("No") != -1 and label == "-1":
            correct_count += 1

print("LLaMA-13B acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/FB13/pred_instructions_llama_raw.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("?")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

print("LLaMA-7B raw acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/FB13/pred_instructions_llama_raw13B.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("?")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

print("LLaMA-13B raw acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/FB13/generated_predictions.txt", "r") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("\"predict\": \"")
        res = line[start_idx + len("\"predict\": \""): -2]
        
        label = labels[i]
        #print(res, label)
        if res.find("Yes") != -1 and label == "1":
            correct_count += 1
        elif res.find("No") != -1 and label == "-1":
            correct_count += 1

print("GLM acc: ", correct_count, 1.0 * correct_count/len(labels))

