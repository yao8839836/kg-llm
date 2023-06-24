
prompts = []
labels = []
with open("data/YAGO3-10/YAGO3_tail_test_human.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        #print(tmp)
        prompts.append(tmp[0])
        labels.append(tmp[1])

correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/pred_instructions_llama_link.csv", "r", encoding="utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(p)
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.rfind(",")
        res = target_line[res_begin_idx + 1:]
        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1
            #print(res, label)


with open("data/YAGO3-10/pred_instructions_llama_link_100.csv", "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-7B acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/pred_instructions_llama_link13B.csv", "r", encoding="utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(p)
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.rfind(",")
        res = target_line[res_begin_idx + 1:]

        
        label = labels[i]
        
        if res.find(label) != -1:
            correct_count += 1
            #print(res, label)


with open("data/YAGO3-10/pred_instructions_llama_link13B_100.csv", "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-13B acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/pred_instructions_llama_link_raw.csv", "r", encoding= "utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        start_idx = target_line.find(prompts[i])
        res = target_line[start_idx + len(prompts[i]):]
        #print(res)
        #print(label)

        label = labels[i]
        if res.find(label) != -1:
            correct_count += 1
            #print(target_line, res, label)

with open("data/YAGO3-10/pred_instructions_llama_link_raw_100.csv", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-7B raw acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/pred_instructions_llama_raw13B.csv", "r", encoding= "utf-8") as f:
    txt = f.read()
    for (i, p) in enumerate(prompts):
        begin_idx = txt.find(p)
        end_idx = txt[begin_idx:].find("\n")
        target_line = txt[begin_idx: begin_idx + end_idx]
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        start_idx = target_line.find(prompts[i])
        res = target_line[start_idx + len(prompts[i]):]
        #print(res)
        #print(label)

        label = labels[i]
        if res.find(label) != -1:
            correct_count += 1
            #print(target_line, res, label)

with open("data/YAGO3-10/pred_instructions_llama_link_raw13B_100.csv", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("LLaMA-13B raw acc: ", correct_count, 1.0 * correct_count/len(labels))


selected_ids =  []
lines_to_write = []
with open("data/YAGO3-10/test_instructions_glm_merge.json", "r",  encoding= "utf-8") as f:
    lines = f.readlines()
    for p in prompts:
        for (idx, line) in enumerate(lines):
            if line.find(p) != -1:
                selected_ids.append(idx)
                lines_to_write.append(line)
                break

#print(selected_ids)

with open("data/YAGO3-10/test_instructions_glm_link_100.json", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)


correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/generated_predictions.txt", "r", encoding= "utf-8") as f:
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
        if res.find(label) != -1:
            correct_count += 1
            #print(line, res, label)


with open("data/YAGO3-10/generated_predictions_100.txt", "w", encoding= "utf-8") as f:
    f.writelines(lines_to_write)

print("GLM acc: ", correct_count, 1.0 * correct_count/len(labels))

correct_count = 0
lines_to_write = []
with open("data/YAGO3-10/raw_generated_predictions_100.txt", "r", encoding= "utf-8") as f:
    lines = f.readlines()

    for (i, target_line) in enumerate(selected_lines):
        target_line = target_line.strip()
        #print(target_line)
        lines_to_write.append(target_line + "\n")
        res_begin_idx = target_line.find("\"predict\": \"")
        res = target_line[res_begin_idx + len("\"predict\": \""): -2]
        #print(res)

        label = labels[i]
        if res.find(label) != -1:
            correct_count += 1
            #print(line, res, label)

print("GLM raw acc: ", correct_count, 1.0 * correct_count/len(labels))