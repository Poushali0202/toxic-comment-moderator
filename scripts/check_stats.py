import json
from collections import Counter
from datasets import load_dataset

LABELS = json.load(open("data/labels.json"))
ds = load_dataset("json", data_files="data/train.jsonl", split="train")

tot = len(ds)
pos_counts = Counter()
for e in ds:
    for l in LABELS:
        if e[l]: pos_counts[l]+=1

print("Total train:", tot)
for l in LABELS:
    print(f"{l:15s}: {pos_counts[l]} ({pos_counts[l]/tot:.2%})")
