# hmm_model.py - probabilities for the HMM (used by q2a, q2b, q2c)
import math
from collections import Counter, defaultdict

A_TRANS = 0.01   # smoothing for transitions
A_EMIT = 0.01    # smoothing for emissions


class HMM:
    def __init__(self, sentences):
        self.tag_count = Counter()
        start, end = Counter(), Counter()
        trans = defaultdict(Counter)       # trans[prev][cur]    (left -> right)
        emit = defaultdict(Counter)        # emit[tag][word]
        word_count = Counter()
        for words, tags in sentences:
            start[tags[0]] += 1
            end[tags[-1]] += 1
            for w, t in zip(words, tags):
                self.tag_count[t] += 1
                emit[t][w] += 1
                word_count[w] += 1
            for a, b in zip(tags, tags[1:]):
                trans[a][b] += 1

        self.tags = sorted(self.tag_count)
        T, N, V = len(self.tags), len(sentences), len(word_count)

        # forward model :  P(first tag), P(tag_i | tag_i-1)
        self.start_f = {t: math.log((start[t] + A_TRANS) / (N + A_TRANS * T)) for t in self.tags}
        self.trans_f = {a: {b: math.log((trans[a][b] + A_TRANS) / (sum(trans[a].values()) + A_TRANS * T))
                            for b in self.tags} for a in self.tags}

        # backward model : P(last tag), P(tag_i | tag_i+1)
        back = defaultdict(Counter)        # back[next][cur]
        for a in trans:
            for b, c in trans[a].items():
                back[b][a] += c
        self.start_b = {t: math.log((end[t] + A_TRANS) / (N + A_TRANS * T)) for t in self.tags}
        self.trans_b = {a: {b: math.log((back[a][b] + A_TRANS) / (sum(back[a].values()) + A_TRANS * T))
                            for b in self.tags} for a in self.tags}

        # emissions P(word | tag); unknown words use words seen only once in train
        self.emit_known = {t: {} for t in self.tags}
        self.emit_unk = {}
        for t in self.tags:
            den = self.tag_count[t] + A_EMIT * (V + 1)
            for w, c in emit[t].items():
                self.emit_known[t][w] = math.log((c + A_EMIT) / den)
            rare = sum(1 for w, c in emit[t].items() if word_count[w] == 1)
            self.emit_unk[t] = math.log((rare + A_EMIT) / den)
            self.unseen_pair = math.log(A_EMIT / den)          # known word, but never with this tag
        self.vocab = set(word_count)
        self._den = {t: self.tag_count[t] + A_EMIT * (V + 1) for t in self.tags}

    def emission(self, word, tag):
        if word not in self.vocab:
            return self.emit_unk[tag]
        return self.emit_known[tag].get(word, math.log(A_EMIT / self._den[tag]))


def viterbi(words, hmm, start, trans):
    """Standard Viterbi. Returns (best_tags, score) where score[i][tag] is the log-prob
    of the best path that ends in `tag` at position i."""
    n = len(words)
    score = [{} for _ in range(n)]
    back = [{} for _ in range(n)]
    for t in hmm.tags:
        score[0][t] = start[t] + hmm.emission(words[0], t)
    for i in range(1, n):
        for t in hmm.tags:
            best_prev = max(hmm.tags, key=lambda p: score[i - 1][p] + trans[p][t])
            score[i][t] = score[i - 1][best_prev] + trans[best_prev][t] + hmm.emission(words[i], t)
            back[i][t] = best_prev
    last = max(hmm.tags, key=lambda t: score[n - 1][t])
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return path[::-1], score
