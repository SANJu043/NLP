import math

sentences = [
    "cat dog animal",
    "cat cat pet",
    "dog pet animal",
    "cat dog pet"
]

vocab = sorted(list(set(" ".join(sentences).split())))
N = len(sentences)

tf_list = []
for s in sentences:
    words = s.split()
    counts = {w: words.count(w) for w in vocab}
    tf_list.append(counts)

idf = {}
for w in vocab:
    df = sum(1 for tf in tf_list if tf[w] > 0)
    idf[w] = math.log10(N / df) if df > 0 else 0

tfidf_list = []
for tf in tf_list:
    vec = {w: tf[w] * idf[w] for w in vocab}
    tfidf_list.append(vec)

def cosine_sim(v1, v2):
    dot = sum(v1[w]*v2[w] for w in vocab)
    norm1 = math.sqrt(sum(v1[w]**2 for w in vocab))
    norm2 = math.sqrt(sum(v2[w]**2 for w in vocab))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

for i in range(N):
    for j in range(i+1, N):
        print(f"Sim(S{i+1}, S{j+1}): {cosine_sim(tfidf_list[i], tfidf_list[j])}")