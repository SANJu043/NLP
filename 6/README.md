# Assignment 6: Advanced Smoothing Techniques for N-gram Language Modeling

This directory implements and evaluates **five advanced smoothing algorithms** across Unigram, Bigram, Trigram, and Quadrigram language models trained on 1,000,000 Telugu sentences ($V = 591,166$ unique vocabulary tokens, $N = 13.3\text{M}$ training tokens).

---

## File Overview

* **`NLP_Assignment6_Smoothing.ipynb`**:
  * Benchmark notebook implementing all five smoothing algorithms, exact count-of-counts caching, runtime tracking, and side-by-side perplexity tables.
* **`U24AI037_NLP_ASS6.ipynb`**:
  * Clean standalone notebook with implementation logic, detailed theoretical observations, and autoregressive text generation.
* **`NLP-Lab-Assignment-6-2026.pdf`**:
  * Problem statement and assignment criteria.

---

## Smoothing Techniques Implemented

1. **Interpolated Smoothing** (Bigram, Trigram, Quadrigram):
   * Linear interpolation of Maximum Likelihood Estimates (MLE) from order $n$ down to unigrams using uniform weights $\lambda = \frac{1}{n}$.
   * Guarantees zero probability is never assigned to any valid token.

2. **Good-Turing Smoothing**:
   * Re-estimates probability mass using frequency-of-frequencies: $c^* = (c + 1) \frac{N_{c+1}}{N_c}$.
   * Unseen events share the total mass of singletons ($N_1 / N$).

3. **Katz Backoff Smoothing**:
   * Uses Good-Turing discounting for observed n-grams.
   * If an n-gram is unseen, backs off recursively to the $(n-1)$-gram distribution, normalized by a precomputed reserved mass factor $\alpha(\text{context})$.

4. **Stupid Backoff Smoothing**:
   * Production-oriented heuristic (Brants et al., Google): uses relative frequency if the n-gram count is positive; otherwise applies a fixed penalty factor $S = 0.4$ and backs off to the $(n-1)$-gram.

5. **Kneser-Ney Smoothing (Interpolated)**:
   * Absolute discounting ($D = 0.75$) at the highest order combined with **continuation counts** at lower orders ($P_{\text{cont}}(w)$).
   * Evaluates how versatile a word is across diverse preceding contexts rather than just its raw frequency.

---

## Experimental Benchmark Results

Perplexity evaluated on **1,000 Development** and **1,000 Test** sentences:

| Smoothing Method | N-gram Model | Dev Perplexity | Test Perplexity |
| :--- | :--- | :---: | :---: |
| **Interpolated** | Bigram | 784.01 | 797.76 |
| | Trigram | 632.01 | 673.71 |
| | Quadrigram | 704.02 | 757.54 |
| **Good-Turing** | Unigram | 3,090.18 | 2,964.93 |
| | Bigram | 1,313.46 | 1,379.93 |
| | Trigram | 8,348.10 | 9,081.00 |
| | Quadrigram | 38,981.25 | 42,909.96 |
| **Katz Backoff** | Unigram | 3,090.18 | 2,964.93 |
| | Bigram | 550.79 | 542.20 |
| | Trigram | **467.51** | **476.13** |
| | Quadrigram | 510.82 | 519.50 |
| **Stupid Backoff** | Unigram | 4,211.87 | 4,260.23 |
| | Bigram | 644.26 | 674.27 |
| | Trigram | 695.70 | 767.46 |
| | Quadrigram | 1,252.93 | 1,401.14 |
| **Kneser-Ney** | Unigram | 8,698.13 | 9,135.02 |
| | Bigram | 847.95 | 921.18 |
| | Trigram | 625.77 | 708.93 |
| | Quadrigram | **616.11** | **703.29** |

---

## Key Analysis & Insights

1. **Top Performers (Katz & Kneser-Ney)**:
   * **Katz Backoff** achieves the lowest perplexity on Trigrams (~467–476) due to precise probability redistribution.
   * **Interpolated Kneser-Ney** achieves the best Quadrigram perplexity (~616–703) because continuation probability prevents frequent unigrams with restricted distribution (e.g., initial particles) from improperly dominating unseen contexts.

2. **Good-Turing Limitation at Higher Orders**:
   * Pure Good-Turing degrades sharply for Trigrams and Quadrigrams (PPL $> 38\text{k}$). Because the number of possible unseen combinations is enormous, sharing the small singleton mass ($N_1 / N$) uniformly gives almost zero probability to test n-grams. Backoff/interpolation architectures are essential.

3. **Stupid Backoff Efficiency**:
   * Despite being an unnormalized score rather than a strict probability distribution, Stupid Backoff yields strong, competitive perplexity with near-zero runtime overhead.

---

## How to Run

Execute either notebook in Jupyter Notebook or Google Colab:

```bash
jupyter notebook NLP_Assignment6_Smoothing.ipynb
# or
jupyter notebook U24AI037_NLP_ASS6.ipynb
```
