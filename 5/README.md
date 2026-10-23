# Assignment 5: Add-K Smoothing (K=0.3) for N-gram Language Models

This directory contains the implementation and benchmark analysis of **Add-$K$ Smoothing ($K = 0.3$)** applied across Unigram, Bigram, Trigram, and Quadrigram language models trained on 1.5M Telugu sentences, including a head-to-head comparison with Laplace smoothing ($K = 1.0$).

---

## File Overview

* **`U24AI037_NLP_ASS5.ipynb`**:
  * Main notebook implementing Add-$K$ probability functions, comparative perplexity benchmarks ($K=1.0$ vs. $K=0.3$), bar chart visualizations, and autoregressive sentence generation.
* **`NLP-Lab-Assignment-5-2026.pdf`**:
  * Assignment problem statement and formal guidelines.

---

## Methodology & Formulation

1. **Probability Estimation**:
   $$\text{Unigram: } P_{\text{Add-}K}(w_i) = \frac{C(w_i) + K}{N + K \cdot V}$$
   $$\text{Higher-Order: } P_{\text{Add-}K}(w_i \mid w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^i) + K}{C(w_{i-n+1}^{i-1}) + K \cdot V}$$
   where $K = 0.3$, $V = 841,362$ unique tokens, and $N$ is the total token count.

2. **Evaluation Metric**:
   * Average sentence perplexity evaluated over **1,000 Development sentences** and **1,000 Test sentences**.

---

## Benchmark Results: Laplace ($K=1.0$) vs. Add-$K$ ($K=0.3$)

| Model Order | Dev ($K = 1.0$) | Dev ($K = 0.3$) | Test ($K = 1.0$) | Test ($K = 0.3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Unigram** | **35,543.35** | 47,203.80 | **26,803.23** | 34,669.62 |
| **Bigram** | 53,965.73 | **39,505.99** | 54,745.23 | **39,153.47** |
| **Trigram** | 265,230.26 | **221,612.81** | 267,207.92 | **221,003.00** |
| **Quadrigram** | 409,650.27 | **375,873.83** | 419,644.07 | **383,470.96** |

---

## Key Observations

1. **Substantial Improvement on Higher Orders**:
   * Tuning $K$ down from $1.0$ to $0.3$ yields significant perplexity reductions across all higher-order models (Bigram drops from ~54k to ~39k; Trigram drops from ~267k to ~221k; Quadrigram drops from ~419k to ~383k).
   * A smaller $K$ dampens the excessive probability mass diverted to the combinatorially vast space of unobserved context combinations.

2. **Unigram Behavior**:
   * For the unigram model, $K=1.0$ yields lower perplexity than $K=0.3$. Because unigrams are densely observed with zero contextual sparsity, a smaller $K$ concentrates more mass on observed frequent words, which slightly penalizes infrequent test tokens.

3. **Generalization Consistency**:
   * Development and test perplexities closely track each other under both schemes, showing that the model does not overfit to the dev split.

---

## How to Run

Open and run the notebook in Jupyter Notebook or Google Colab:

```bash
jupyter notebook U24AI037_NLP_ASS5.ipynb
```
