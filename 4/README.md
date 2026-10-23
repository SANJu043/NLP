# Assignment 4: Statistical N-gram Language Modeling with Laplace (Add-1) Smoothing

This directory contains the implementation of **Unigram, Bigram, Trigram, and Quadrigram Language Models** trained on a large Telugu corpus and smoothed using **Laplace (Add-1)** smoothing.

---

## File Overview

* **`U24AI037_NLP_ASS4.ipynb`**:
  * Complete end-to-end notebook implementing data loading, cleaning, n-gram extraction, Laplace probability estimation, perplexity evaluation, and autoregressive sentence generation.
* **`NLP-Lab-Assignment-4-2026.pdf`**:
  * Assignment problem statement and formal specifications.

---

## Pipeline & Implementation

1. **Dataset Sampling & Splits**:
   * Sampled **1,500,000 sentences** from `telugu_tokenized_full.parquet` using DuckDB.
   * Split into:
     * **Training set**: 1,498,000 sentences
     * **Development set**: 1,000 sentences
     * **Test set**: 1,000 sentences
   * Cleaned using regex to remove stray Latin characters and normalize whitespaces.
   * **Vocabulary size ($V$)**: 841,362 unique tokens.

2. **N-gram Construction**:
   * Extracted n-grams with sentence boundary padding (`<s>` and `</s>`) for orders $n \in \{1, 2, 3, 4\}$.
   * Stored frequency tables for both n-gram tuples and context tuples.

3. **Laplace (Add-1) Probability Formulation**:
   $$\text{Unigram: } P_{\text{Add-1}}(w_i) = \frac{C(w_i) + 1}{N + V}$$
   $$\text{Higher-Order: } P_{\text{Add-1}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^i) + 1}{C(w_{i-n+1}^{i-1}) + V}$$
   where $N$ is the total token count and $V$ is the vocabulary size.

4. **Perplexity Evaluation**:
   $$\text{Perplexity} = \exp\left(-\frac{1}{N_{\text{tokens}}} \sum_{i=1}^{N_{\text{tokens}}} \ln P(w_i \mid \text{context})\right)$$

5. **Sentence Generation**:
   * Stochastically samples the next token conditioned on prior $(n-1)$ context tokens until the end-of-sentence token (`</s>`) or word limit is reached.

---

## Key Findings & Observations

* **Aggressive Over-Smoothing in Higher Orders**:
  * Under Add-1 smoothing, higher-order models (Trigram and Quadrigram) exhibit higher perplexity than lower orders.
  * *Reason*: With a large vocabulary ($V \approx 841\text{k}$), the number of possible n-gram contexts grows exponentially. Adding $+1$ to all unseen combinations distributes a disproportionately large amount of probability mass away from the true, seen text.
* **Motivation for Assignment 5**:
  * This severe over-smoothing penalty directly motivated transitioning from Add-1 to fractional Add-$K$ smoothing ($K=0.3$) to preserve probability mass for observed contexts.
* **Generalization**:
  * Development and test set perplexities remain closely aligned, confirming a representative data split without distribution shift.

---

## How to Run

Open and execute the notebook in Jupyter Notebook or Google Colab:

```bash
jupyter notebook U24AI037_NLP_ASS4.ipynb
```
