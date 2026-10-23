"""
Noisy-channel spell corrector
    w* = argmax_w  P(w) * P(x | w)

  P(w)    : unigram language model from count_unigram.txt (add-one smoothed)
  P(x|w)  : built from count_1edit.txt ("correct|typed  count") using the
            Kernighan/Church/Gale formulation
                  P(op) = eps * count(correct|typed) / count(correct substring)
            where count(correct substring) is a char / bigram count taken from
            the weighted unigram corpus ('>' = start of word).
            Multi-edit errors = product of per-edit probabilities.
  Candidates : all vocabulary words within Damerau edit distance <= 2
            (found through a SymSpell-style delete index); the best alignment
            (Viterbi/Damerau DP) gives the channel probability.
  P(no error): a typed word that is itself in the vocabulary is also a
            candidate with channel probability (1 - eps).
"""
import math, collections, itertools, random, sys, json
import os
D = os.path.dirname(os.path.abspath(__file__)) + os.sep   # files expected in the same folder as this script
VOCAB_SIZE = 100000

# ------------------------------------------------------------------ data
uni = {}
for l in open(D+'count_unigram.txt', encoding='latin-1'):
    p = l.split()
    if len(p) == 2: uni[p[0]] = int(p[1])
N = sum(uni.values())
vocab_list = sorted(uni, key=uni.get, reverse=True)[:VOCAB_SIZE]
vocab = set(vocab_list)
V = len(uni)
def logP(w): return math.log((uni.get(w, 0)+1)/(N+V))

edits = {}
for l in open(D+'count_1edit.txt', encoding='latin-1'):
    k, c = l.rstrip('\n').rsplit('\t', 1)
    edits[k] = int(c)
E_TOTAL = sum(edits.values())
ctx = collections.Counter()
for w, c in uni.items():
    s = '>'+w
    for i in range(len(s)):
        ctx[s[i]] += c
        if i+1 < len(s): ctx[s[i:i+2]] += c
C1 = sum(v for k, v in ctx.items() if len(k) == 1)
SCALE = C1/E_TOTAL            # makes sum over edit types of count/ctx*SCALE ~ 1
EPS = 0.01                    # P(an edit occurs); tuned on a dev split

def edit_cost(c, t):
    """-log P(typed t | correct c) for ONE elementary edit (excluding eps)."""
    p = (edits.get(c+'|'+t, 0)+1) / (ctx.get(c, 0)+1) * SCALE
    return -math.log(min(p, 1.0))

# -------------------------------------------------- SymSpell delete index
def deletes(w, d=2):
    out = {w}; frontier = {w}
    for _ in range(d):
        nxt = set()
        for x in frontier:
            for i in range(len(x)):
                nxt.add(x[:i]+x[i+1:])
        out |= nxt; frontier = nxt
    return out
index = collections.defaultdict(list)
for w in vocab_list:
    for d in deletes(w): index[d].append(w)

# ------------------------------------------------ alignment / channel DP
_cache = {}
def channel_cost(c, t):
    """min over alignments of sum of -log P(edit); c=correct, t=typed."""
    key = (c, t)
    if key in _cache: return _cache[key]
    C, T = '>'+c, '>'+t
    n, m = len(C), len(T)
    INF = 1e9
    dp = [[INF]*m for _ in range(n)]
    dp[0][0] = 0.0
    for i in range(n):
        for j in range(m):
            cur = dp[i][j]
            if cur >= INF: continue
            if i+1 < n and j+1 < m:                       # match / substitute
                x = cur if C[i+1] == T[j+1] else cur+edit_cost(C[i+1], T[j+1])
                if x < dp[i+1][j+1]: dp[i+1][j+1] = x
            if j+1 < m:                                    # typed has extra char
                x = cur+edit_cost(C[i], C[i]+T[j+1])
                if x < dp[i][j+1]: dp[i][j+1] = x
            if i+1 < n:                                    # typed dropped a char
                x = cur+edit_cost(C[i]+C[i+1], C[i])
                if x < dp[i+1][j]: dp[i+1][j] = x
            if i+2 < n and j+2 < m and C[i+1] == T[j+2] and C[i+2] == T[j+1] and C[i+1] != C[i+2]:
                x = cur+edit_cost(C[i+1]+C[i+2], T[j+1]+T[j+2])   # transposition
                if x < dp[i+2][j+2]: dp[i+2][j+2] = x
    r = dp[n-1][m-1]
    _cache[key] = r
    return r

def candidates(t, maxd=2):
    t = t.lower()
    found = set()
    for d in deletes(t, maxd):
        found.update(index.get(d, ()))
    return found

def score_all(t, eps=EPS, lam=1.0):
    t = t.lower()
    sc = {}
    for w in candidates(t):
        if w == t: continue
        cost = channel_cost(w, t)
        if cost > 1e8: continue
        n_edits = None
        sc[w] = lam*logP(w) - cost + _edits_log_eps(w, t, cost, eps)
    if t in uni:
        sc[t] = lam*logP(t) + math.log(1-eps)
    return sc

def _edits_log_eps(w, t, cost, eps):
    # number of edits ~ Damerau distance; approximate by len-diff/DP-free rule
    return osa(w, t)*math.log(eps)

def osa(a, b):
    la, lb = len(a), len(b)
    d = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): d[i][0] = i
    for j in range(lb+1): d[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cst = a[i-1] != b[j-1]
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cst)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2]+1)
    return d[la][lb]

def correct(t, eps=EPS, lam=1.0):
    sc = score_all(t, eps, lam)
    return max(sc, key=sc.get) if sc else t.lower()

def load_tests():
    tests = []
    for l in open(D+'spell-errors.txt', encoding='latin-1'):
        if ':' not in l: continue
        w, rest = l.rstrip('\n').split(':', 1)
        for e in rest.split(','):
            e = e.strip()
            if not e: continue
            n = 1
            if '*' in e: e, n = e.split('*'); n = int(n)
            tests.append((e.lower(), w.strip().lower(), n))
    return tests

def accuracy(tests, **kw):
    ok = okw = totw = 0
    wrong = []
    for t, g, n in tests:
        p = correct(t, **kw); totw += n
        if p == g: ok += 1; okw += n
        else: wrong.append((t, g, p))
    return ok/len(tests), okw/totw, wrong

if __name__ == '__main__':
    # usage:  python spell_checker_local.py            -> evaluate accuracy on spell-errors.txt
    #         python spell_checker_local.py raning yello  -> correct the given words
    if len(sys.argv) > 1:
        for w in sys.argv[1:]:
            print(w, '->', correct(w, eps=0.2))
        sys.exit()
    tests = load_tests()
    ok = okw = totw = 0
    wrong = []
    for k, (t, g, n) in enumerate(tests, 1):
        p = correct(t, eps=0.2, lam=1.0)
        totw += n
        if p == g: ok += 1; okw += n
        else: wrong.append((t, g, p))
        if k % 5000 == 0: print(f'{k}/{len(tests)}  running accuracy {ok/k:.4f}', flush=True)
    print(f'pairs: {len(tests)}  accuracy: {ok/len(tests):.4f}   weighted by *n: {okw/totw:.4f}')
    random.seed(0)
    print('sample errors (typo, gold, predicted):')
    for x in random.sample(wrong, 15): print('  ', x)