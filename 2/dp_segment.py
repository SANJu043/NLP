import json
import math

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["word_counts"], data["test_cases"]


def build_log_probs(word_counts):
    total_count = sum(word_counts.values())
    log_probs = {w: math.log(c / total_count) for w, c in word_counts.items()}
    return log_probs, total_count

def word_score(word, log_probs, total_count):
    if word in log_probs:
        return log_probs[word]
    return math.log(1.0 / (total_count * (10 ** len(word))))

def dp_segment(text, log_probs, total_count, max_word_len):
    n = len(text)
    best_score = [-math.inf] * (n + 1)
    best_split = [None] * (n + 1)
    best_score[0] = 0.0

    for i in range(1, n + 1):
        start = max(0, i - max_word_len)
        for j in range(start, i):
            if best_score[j] == -math.inf:
                continue
            word = text[j:i]
            score = best_score[j] + word_score(word, log_probs, total_count)
            if score > best_score[i]:
                best_score[i] = score
                best_split[i] = j

    words = []
    i = n
    while i > 0:
        j = best_split[i]
        words.append(text[j:i])
        i = j
    words.reverse()
    return words


def levenshtein(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def evaluate(test_cases, log_probs, total_count, max_word_len):
    total = len(test_cases)
    exact_matches = 0
    word_edit_distances = []
    char_edit_distances = []
    for case in test_cases:
        pred_words = dp_segment(case["input"], log_probs, total_count, max_word_len)
        gt_words = case["ground_truth"].split()
        pred_str = " ".join(pred_words)
        gt_str = case["ground_truth"]
        if pred_str == gt_str:
            exact_matches += 1
        word_edit_distances.append(levenshtein(pred_words, gt_words))
        char_edit_distances.append(levenshtein(pred_str, gt_str))
    accuracy = (exact_matches / total) * 100
    avg_word_ed = sum(word_edit_distances) / total
    avg_char_ed = sum(char_edit_distances) / total
    return accuracy, avg_word_ed, avg_char_ed


def main():
    word_counts, test_cases = load_data("text_segmentation_dataset.json")
    log_probs, total_count = build_log_probs(word_counts)
    max_word_len = max(len(w) for w in word_counts)

    accuracy, avg_word_ed, avg_char_ed = evaluate(test_cases, log_probs, total_count, max_word_len)

    print("Dynamic Programming (Max Log-Probability) Segmentation")
    print(f"Test cases evaluated : {len(test_cases)}")
    print(f"Exact-match Accuracy : {accuracy:.2f}%")
    print(f"Avg Word-level Edit Distance : {avg_word_ed:.4f}")
    print(f"Avg Char-level Edit Distance : {avg_char_ed:.4f}")

    print("\nSample predictions:")
    for case in test_cases[:5]:
        pred = " ".join(dp_segment(case["input"], log_probs, total_count, max_word_len))
        print(f"  Input : {case['input']}")
        print(f"  Pred  : {pred}")
        print(f"  Truth : {case['ground_truth']}\n")


if __name__ == "__main__":
    main()