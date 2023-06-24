
prompts = []
labels = []

with open("data/FB13/FB13_test_human.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        prompts.append(tmp[1])
        labels.append(tmp[0])

correct_count = 0
lines_to_write = []
with open("data/FB13/pred_instructions_llama.csv", "r") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("?,")
        res = target_line[res_begin_idx + 3: -1]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

with open("data/FB13/pred_instructions_llama_100.csv", "w") as f:
    f.writelines(lines_to_write)

print("LLaMA-7B acc: ", correct_count, 1.0 * correct_count/len(labels))


correct_count = 0
lines_to_write = []
with open("data/FB13/pred_instructions_llama13b.csv", "r") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("?,")
        res = target_line[res_begin_idx + 3: -1]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

with open("data/FB13/pred_instructions_llama13b_100.csv", "w") as f:
    f.writelines(lines_to_write)

print("LLaMA-13B acc: ", correct_count, 1.0 * correct_count/len(labels))


correct_count = 0
lines_to_write = []
with open("data/FB13/pred_instructions_llama_raw.csv", "r", encoding= "utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("?,")
        res = target_line[res_begin_idx + 3: -1]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

with open("data/FB13/pred_instructions_llama_raw_100.csv", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-7B raw acc: ", correct_count, 1.0 * correct_count/len(labels))


correct_count = 0
lines_to_write = []
with open("data/FB13/pred_instructions_llama_raw13b.csv", "r", encoding= "utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("?,")
        res = target_line[res_begin_idx + 3: -1]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

with open("data/FB13/pred_instructions_llama_raw13b_100.csv", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-13B raw acc: ", correct_count, 1.0 * correct_count/len(labels))

selected_ids =  []
lines_to_write = []
with open("data/FB13/test_instructions_glm.json", "r") as f:
    lines = f.readlines()
    for p in prompts:
        for (idx, line) in enumerate(lines):
            if line.find(p) != -1:
                selected_ids.append(idx)
                lines_to_write.append(line)
                break

print(selected_ids)

with open("data/FB13/test_instructions_glm_100.json", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

correct_count = 0
lines_to_write = []
with open("data/FB13/generated_predictions.txt", "r") as f:
    lines = f.readlines()
    selected_lines = [lines[x] for x in selected_ids]
    for (i, target_line) in enumerate(selected_lines):
        target_line = target_line.strip()
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("\"predict\": \"")
        res = target_line[res_begin_idx + len("\"predict\": \""): -2]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

with open("data/FB13/generated_predictions_100.txt", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("GLM acc: ", correct_count, 1.0 * correct_count/len(labels))


correct_count = 0
lines_to_write = []
with open("data/FB13/fb13_raw_generated_predictions.txt", "r", encoding= "utf-8") as f:
    lines = f.readlines()

    for (i, target_line) in enumerate(lines):
        target_line = target_line.strip()
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("\"predict\": \"")
        res = target_line[res_begin_idx + len("\"predict\": \""): -2]
        #print(res)

        label = labels[i]
        if (res.find("Yes,") != -1 or res.find(" yes") != -1) and label == "1":
            correct_count += 1
            #print(line, res, label)
        elif (res.find("No") != -1 or res.find("not") != -1 or res.find("n't") != -1 or res.find("no") != -1) and label == "-1":
            correct_count += 1

print("GLM raw acc: ", correct_count, 1.0 * correct_count/len(labels))