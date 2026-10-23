# common.py - data loading + F1 evaluation (shared by all question files)
import json, os

DATA_DIR = "Hindi-POS-Data"      # folder with the 3 files
RESULT_DIR = "results"


def read_file(path):
    """Returns a list of sentences. Each sentence = (list_of_words, list_of_tags).
    Handles:  (a) 'word TAG' on each line, blank line between sentences
              (b) one sentence per line with tokens like word_TAG or word/TAG"""
    lines = [l.strip() for l in open(path, encoding="utf-8-sig")]
    tokens = [t for l in lines[:300] for t in l.split()]
    inline = sum(("_" in t or "/" in t) for t in tokens) > 0.9 * max(1, len(tokens))

    sentences = []
    if inline:
        for line in lines:
            words, tags = [], []
            for tok in line.split():
                sep = "_" if "_" in tok else "/"
                w, t = tok.rsplit(sep, 1)
                words.append(w)
                tags.append(t)
            if words:
                sentences.append((words, tags))
    else:
        words, tags = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                words.append(parts[0])
                tags.append(parts[-1])
            elif words:                      # blank line = end of sentence
                sentences.append((words, tags))
                words, tags = [], []
        if words:
            sentences.append((words, tags))
    return sentences


def load_all():
    names = {"train": "hindi-train-pos.txt", "dev": "hindi-dev-pos.txt", "test": "hindi-test-pos.txt"}
    return {k: read_file(os.path.join(DATA_DIR, v)) for k, v in names.items()}


def f1_report(gold, pred):
    """Per-tag precision / recall / F1 + accuracy."""
    report = {}
    for tag in sorted(set(gold) | set(pred)):
        tp = sum(1 for g, p in zip(gold, pred) if g == tag and p == tag)
        fp = sum(1 for g, p in zip(gold, pred) if g != tag and p == tag)
        fn = sum(1 for g, p in zip(gold, pred) if g == tag and p != tag)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        report[tag] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    report["_accuracy"] = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return report


def evaluate_and_save(model_name, predict_fn, data):
    """Runs predict_fn(words)->tags on dev and test, prints F1 per tag, saves json."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {}
    for split in ("dev", "test"):
        gold, pred = [], []
        for words, tags in data[split]:
            gold += tags
            pred += predict_fn(words)
        out[split] = f1_report(gold, pred)

        print(f"\n{model_name} - {split.upper()}   accuracy = {out[split]['_accuracy']:.4f}")
        print(f"{'TAG':<10}{'P':>8}{'R':>8}{'F1':>8}{'count':>8}")
        for tag, r in out[split].items():
            if tag != "_accuracy":
                print(f"{tag:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['support']:>8}")

    with open(os.path.join(RESULT_DIR, model_name + ".json"), "w") as f:
        json.dump(out, f, indent=1)
