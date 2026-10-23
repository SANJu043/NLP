sentences = [
    "I like NLP",
    "I like AI",
    "You like NLP",
    "You love AI",
    "We love NLP"
]

tokens = [s.split() for s in sentences]
bigrams = []
for sent in tokens:
    for i in range(len(sent)-1):
        bigrams.append((sent[i], sent[i+1]))

from collections import Counter
bi_counts = Counter(bigrams)
total_unique_bigrams = len(bi_counts)

preceding_words = {w: set() for sent in tokens for w in sent}
for (w1, w2) in bi_counts:
    preceding_words[w2].add(w1)

def p_continuation(w2):
    return len(preceding_words[w2]) / total_unique_bigrams if total_unique_bigrams > 0 else 0

def kneser_ney(w2, w1, d=0.5):
    c_w1_w2 = bi_counts[(w1, w2)]
    sum_c_w1 = sum(count for (a, b), count in bi_counts.items() if a == w1)
    unique_following = sum(1 for (a, b) in bi_counts if a == w1)
    
    if sum_c_w1 == 0:
        return p_continuation(w2)
    
    first_term = max(c_w1_w2 - d, 0) / sum_c_w1
    lambda_w1 = (d / sum_c_w1) * unique_following
    return first_term + (lambda_w1 * p_continuation(w2))

print("P_KN(AI | love):", kneser_ney("AI", "love"))
print("P_KN(NLP | like):", kneser_ney("NLP", "like"))