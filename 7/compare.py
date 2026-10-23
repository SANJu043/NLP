# compare.py - run AFTER q1, q2a, q2b, q2c. Compares all algorithms and plots F1 per tag.
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULT_DIR

MODELS = ["naive_bayes", "hmm_forward_viterbi", "hmm_backward_viterbi", "hmm_forward_backward_viterbi"]
results = {m: json.load(open(os.path.join(RESULT_DIR, m + ".json")))
           for m in MODELS if os.path.exists(os.path.join(RESULT_DIR, m + ".json"))}

for split in ("dev", "test"):
    first = next(iter(results.values()))[split]
    tags = sorted((t for t in first if t != "_accuracy"), key=lambda t: -first[t]["support"])

    # table
    print(f"\n{split.upper()}  F1 per tag")
    print(f"{'TAG':<10}" + "".join(f"{m[:18]:>20}" for m in results))
    for t in tags:
        print(f"{t:<10}" + "".join(f"{r[split].get(t, {'f1': 0})['f1']:>20.3f}" for r in results.values()))
    print(f"{'ACCURACY':<10}" + "".join(f"{r[split]['_accuracy']:>20.3f}" for r in results.values()))

    # bar chart: F1 of every tag for every algorithm
    width = 0.8 / len(results)
    plt.figure(figsize=(14, 5.5))
    for k, (m, r) in enumerate(results.items()):
        xs = [i + k * width for i in range(len(tags))]
        plt.bar(xs, [r[split].get(t, {"f1": 0})["f1"] for t in tags], width, label=m)
    plt.xticks([i + 0.4 - width / 2 for i in range(len(tags))], tags, rotation=60, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.title(f"F1-score per tag ({split} set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"f1_per_tag_{split}.png"), dpi=150)
    plt.close()
print("\nCharts saved in", RESULT_DIR)
