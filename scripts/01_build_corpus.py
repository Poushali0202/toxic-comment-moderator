# scripts/01_build_corpus.py
import re, unicodedata, random, json
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from datasets import Features, Value

HEADS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")              # zero-width
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_multi_label(example, mapping):
    """Apply dataset-specific mapping → unified multi-label heads."""
    y = {h: 0 for h in HEADS}
    y.update(mapping(example))
    # Try common text keys; fallback to 'text' if already set
    text = example.get("text") or example.get("comment_text") or example.get("content") or example.get("tweet")
    example["text"] = normalize_text(text or "")
    example.update(y)
    return example

def keep_useful(e):
    return bool(e["text"]) and 3 <= len(e["text"]) <= 1024

def keep_only_needed(e):
    return {
        "text": e["text"],
        **{h: int(e[h]) for h in HEADS}
    }

def dedupe(ds: Dataset) -> Dataset:
    seen, rows = set(), []
    for e in ds:
        t = e["text"].lower()
        if t in seen: 
            continue
        seen.add(t)
        rows.append(e)
    return Dataset.from_list(rows)

def main():
    random.seed(42)

    datasets = []

    # 1) Civil Comments (continuous toxicity)
    print("Downloading: civil_comments")
    cc = load_dataset("civil_comments")
    def map_cc(e):
        tox = 1 if e.get("toxicity", 0) >= 0.5 else 0
        return {"toxic": tox}
    cc_all = cc["train"].map(lambda e: to_multi_label({"text": e["text"], **e}, map_cc))
    datasets.append(cc_all)

    # 2) HateXplain (0=normal, 1=abusive, 2=hate)
    print("Downloading: hatexplain")
    # hx = load_dataset("hatexplain", trust_remote_code=True)
    # def map_hx(e):
    #     lab = e["label"]
    #     m = {}
    #     if lab == 2: 
    #         m["identity_hate"] = 1; m["toxic"]=1
    #     if lab == 1: 
    #         m["insult"]=1; m["toxic"]=1
    #     return m
    # hx_train = hx["train"].map(lambda e: to_multi_label({"text":" ".join(e["post_tokens"])}, map_hx))
    # hx_val   = hx["validation"].map(lambda e: to_multi_label({"text":" ".join(e["post_tokens"])}, map_hx))
    hx = load_dataset("hatexplain", trust_remote_code=True)

# Combine multiple annotator labels by majority vote
    def get_majority_label(annotator_labels):
        flat = [lab for sublist in annotator_labels for lab in sublist if lab in [0, 1, 2]]
        if not flat:
            return 0
        return max(set(flat), key=flat.count)

    def map_hx(e):
        lab = get_majority_label(e.get("annotator_labels", []))
        m = {}
        if lab == 2:  # hate
            m["identity_hate"] = 1; m["toxic"] = 1
        elif lab == 1:  # abusive
            m["insult"] = 1; m["toxic"] = 1
        return m

    hx_train = hx["train"].map(lambda e: to_multi_label({"text": " ".join(e["post_tokens"])}, map_hx))
    hx_val   = hx["validation"].map(lambda e: to_multi_label({"text": " ".join(e["post_tokens"])}, map_hx))
    datasets += [hx_train, hx_val]

    # 3) ETHOS (binary multi-label)
    print("Downloading: ethos")
    eth = load_dataset("ethos", "binary", trust_remote_code=True)["train"]
    def map_eth(e):
        m = {}
        if e.get("insult"): m["insult"]=1; m["toxic"]=1
        if e.get("hatespeech"): m["identity_hate"]=1; m["toxic"]=1
        if e.get("obscene"): m["obscene"]=1; m["toxic"]=1
        if e.get("violence"): m["threat"]=1; m["toxic"]=1
        return m
    eth = eth.map(lambda e: to_multi_label({"text": e["text"]}, map_eth))
    datasets.append(eth)

    # 4) OLID (twitter offensive: 1=offensive)
    print("Downloading: OLID (Offensive Language Identification)")
    try:
        olid = load_dataset("tuner007/OffensEval2020")["train"]
    except Exception:
        print("⚠️ OLID not found — skipping this dataset.")
        olid = None

    if olid:
        def map_olid(e):
            m={}
            # label: 0=not offensive, 1=offensive
            if e.get("label") == 1 or e.get("subtask_a") == "OFF":
                m["toxic"]=1; m["insult"]=1
            return m

        # use "tweet" or "text" column depending on dataset version
        text_col = "tweet" if "tweet" in olid.column_names else "text"
        olid = olid.map(lambda e: to_multi_label({"text": e[text_col]}, map_olid))
        datasets.append(olid)

    # Combine

    print("Combining datasets with unified features...")

    # Define the common schema for alignment
    features = Features({
        "text": Value("string"),
        "toxic": Value("int64"),
        "severe_toxic": Value("int64"),
        "obscene": Value("int64"),
        "threat": Value("int64"),
        "insult": Value("int64"),
        "identity_hate": Value("int64"),
    })

    # Apply same schema to each dataset
    datasets_aligned = []
    for d in datasets:
        # Cast only if it has matching columns
        cols_to_keep = [c for c in d.column_names if c in features]
        d = d.remove_columns([c for c in d.column_names if c not in features])
        d = d.cast(features)
        datasets_aligned.append(d)

    # Now concatenate safely
    ds = concatenate_datasets(datasets_aligned)


    # Clean + keep only needed fields
    ds = ds.filter(keep_useful)
    ds = ds.map(keep_only_needed, remove_columns=[c for c in ds.column_names if c not in (["text"]+HEADS)])

    # Deduplicate
    print("Deduplicating...")
    ds = dedupe(ds)
    print(f"Total after dedupe: {len(ds)}")

    # Balance: cap negatives so they don't dominate (3x positives)
    pos = [e for e in ds if sum(e[h] for h in HEADS) > 0]
    neg = [e for e in ds if sum(e[h] for h in HEADS) == 0]
    cap_neg = min(len(neg), 3*len(pos))
    neg = random.sample(neg, k=cap_neg)
    rows = neg + pos
    random.shuffle(rows)
    ds = Dataset.from_list(rows)
    print(f"After balancing: {len(ds)} (pos={len(pos)}, neg_kept={len(neg)})")

    # Split train/val/test = 81% / 9% / 10%
    train_val, test = train_test_split(rows, test_size=0.10, random_state=42)
    train, val = train_test_split(train_val, test_size=0.10, random_state=42)

    Dataset.from_list(train).to_json("data/train.jsonl", lines=True)
    Dataset.from_list(val).to_json("data/val.jsonl", lines=True)
    Dataset.from_list(test).to_json("data/test.jsonl", lines=True)

    with open("data/labels.json","w") as f:
        json.dump(HEADS, f)

    print("Saved files:")
    print(" data/train.jsonl")
    print(" data/val.jsonl")
    print(" data/test.jsonl")
    print(" data/labels.json")

if __name__ == "__main__":
    main()
