import json
preds = []
with open("data/YAGO3-10/kgt5_merge_predictions.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        preds.append(line.strip())

labels = []
with open("data/YAGO3-10/test_instructions_glm_merge.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        target = json.loads(line.strip())["response"]
        #print(target)
        labels.append(target)

correct_count = 0
for (i, pred) in enumerate(preds):
    if pred == labels[i]:
        correct_count += 1

hits1 = 1.0 * correct_count / len(labels)
print("KGT5 YAGO3-10 link hits@1: ", hits1)

preds = []
with open("data/YAGO3-10/kgt5_rel_predictions.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        preds.append(line.strip())

labels = []
with open("data/YAGO3-10/test_instructions_glm_rel.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        target = json.loads(line.strip())["response"]
        #print(target)
        labels.append(target)

correct_count = 0
for (i, pred) in enumerate(preds):
    if pred == labels[i]:
        correct_count += 1

hits1 = 1.0 * correct_count / len(labels)
print("KGT5 YAGO3-10 relation hits@1: ", hits1)