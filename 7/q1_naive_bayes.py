# Q1 - Naive Bayes POS tagger
# Features: length (MORE/LESS), word, previous word, next word, previous tag
import math
from collections import Counter, defaultdict
from common import load_all, evaluate_and_save

ALPHA = 0.1   # add-alpha smoothing


def get_features(words, i, prev_tag):
    return {
        "length": "MORE" if len(words[i]) > 4 else "LESS",
        "word": words[i],
        "prev_word": words[i - 1] if i > 0 else "<START>",
        "next_word": words[i + 1] if i < len(words) - 1 else "<END>",
        "prev_tag": prev_tag,
    }


# ---------- training: count everything from the train data ----------
def train(sentences):
    tag_count = Counter()
    feat_count = defaultdict(Counter)      # feat_count[(feature, value)][tag]
    values_seen = defaultdict(set)         # distinct values of each feature
    for words, tags in sentences:
        for i, tag in enumerate(tags):
            prev_tag = tags[i - 1] if i > 0 else "<START>"
            tag_count[tag] += 1
            for f, v in get_features(words, i, prev_tag).items():
                feat_count[(f, v)][tag] += 1
                values_seen[f].add(v)
    return tag_count, feat_count, values_seen


# ---------- prediction ----------
def predict(words):
    result, prev_tag = [], "<START>"
    total = sum(tag_count.values())
    for i in range(len(words)):
        feats = get_features(words, i, prev_tag)
        best_tag, best_score = None, -math.inf
        for tag in tag_count:
            score = math.log(tag_count[tag] / total)                    # P(tag)
            for f, v in feats.items():
                if (f, v) not in feat_count:                            # unseen value -> skip
                    continue
                num = feat_count[(f, v)][tag] + ALPHA
                den = tag_count[tag] + ALPHA * len(values_seen[f])
                score += math.log(num / den)                            # P(feature | tag)
            if score > best_score:
                best_tag, best_score = tag, score
        result.append(best_tag)
        prev_tag = best_tag          # use predicted tag as "previous tag" for next word
    return result


if __name__ == "__main__":
    data = load_all()
    tag_count, feat_count, values_seen = train(data["train"])
    evaluate_and_save("naive_bayes", predict, data)
