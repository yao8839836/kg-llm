ent2txt = {}
labels = []
with open("data/WN11/test.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        labels.append(tmp[3])

correct_count = 0
with open("data/WN11/pred_instructions_llama.csv", "r") as f:
    lines = f.readlines()
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

print("LLaMA-7B (tuned) acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN11/pred_instructions_llama13b.csv", "r") as f:
    lines = f.readlines()
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

print("LLaMA-13B (tuned) acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN11/pred_instructions_llama_raw13b.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines[1:]):
        line = line.strip()
        start_idx = line.find("\"")
        res = line[start_idx + 1: -1]
        
        label = labels[i]
        #print(res, label)
        # if res.find("Yes") != -1 and label == "1":
        #     correct_count += 1
        # elif res.find("No") != -1 and label == "-1":
        #     correct_count += 1
        if (res.find("Yes") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1
            
print("LLaMA-13B (raw) acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN11/pred_instructions_llama_raw.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for (i, line) in enumerate(lines):
        line = line.strip()
        start_idx = line.find("?")
        res = line[start_idx + 1:]
        
        label = labels[i]
        
        if (res.find("Yes") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1
            #print(res, label)

print("LLaMA (original) acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
with open("data/WN11/generated_predictions.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()

        label_begin_idx = line.find("\"labels\": \"")
        label_end_idx = line.find("\",")
        label = line[label_begin_idx + len("\"labels\": \""): label_end_idx]

        pred_begin_idx = line.find("\"predict\": \"")
        pred_end_idx = line.find("\"}")
        pred = line[pred_begin_idx + len("\"predict\": \""): pred_end_idx]
        #print(label, pred)
        if pred.find("Yes") != -1 and label.find("Yes") != -1:
            correct_count += 1
        elif pred.find("No") != -1 and label.find("No") != -1:
            correct_count += 1
print("GLM acc: ", correct_count, 1.0 * correct_count/len(labels))

