import json

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
        return data["word_counts"], data["test_cases"]
    
def greedy_segment_ltor(text, vocab, max_word_len):
    words = []
    i = 0
    n = len(text)
    while i < n:
        flag = False
        upper = min(max_word_len, n - i)
        for leng in range(upper, 0, -1):
            curr = text[i:i + leng]
            if curr in vocab:
                words.append(curr)
                i += leng
                flag = True
                break
        if not flag:
            words.append(text[i])
            i += 1
    return words

def greedy_segment_rtol(text, vocab, max_word_len):
    words = []
    i = len(text)
    while i > 0:
        flag = False
        upper = min(max_word_len, i)
        for leng in range(upper, 0, -1):
            curr = text[i - leng:i]
            if curr in vocab:
                words.append(curr)
                i -= leng
                flag = True
                break
        if not flag:
            words.append(text[i - 1])
            i -= 1
    words.reverse()
    return words

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def test(test_cases, vocab, max_word_len, greedy_segment):
    total = len(test_cases)
    exact_matches = 0
    word_edit_distances = []
    char_edit_distances = []
    for case in test_cases:
        pred = greedy_segment(case["input"], vocab, max_word_len)
        pred_str = " ".join(pred)
        gt_str = case["ground_truth"]
        gt_words = gt_str.split()
        if pred_str == gt_str:
            exact_matches += 1
        word_edit_distances.append(levenshtein(pred, gt_words))
        char_edit_distances.append(levenshtein(pred_str, gt_str))
    accuracy = (exact_matches / total) * 100
    avg_word_ed = sum(word_edit_distances) / total
    avg_char_ed = sum(char_edit_distances) / total
    return accuracy, avg_word_ed, avg_char_ed

# Main Logic
word_counts, test_cases = load_data("text_segmentation_dataset.json")
vocab = set(word_counts.keys())
max_word_len = max(len(w) for w in vocab)

func = [
    (greedy_segment_ltor, "Left To Right"),
    (greedy_segment_rtol, "Right To Left"),
]

for greedy_segment, name in func:
    accuracy, avg_word_ed, avg_char_ed = test(test_cases, vocab, max_word_len, greedy_segment)
    print(f"Greedy Longest-Match Segmentation: {name}")
    print(f"Test cases evaluated : {len(test_cases)}")
    print(f"Exact-match Accuracy : {accuracy:.2f}%")
    print(f"Avg Word-level Edit Distance : {avg_word_ed:.4f}")
    print(f"Avg Char-level Edit Distance : {avg_char_ed:.4f}")
    print("\nSample predictions:")
    for case in test_cases[:5]:
        pred = " ".join(greedy_segment(case["input"], vocab, max_word_len))
        print(f"  Input : {case['input']}")
        print(f"  Pred  : {pred}")
        print(f"  Truth : {case['ground_truth']}\n")
    
