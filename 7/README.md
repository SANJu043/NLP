# Assignment 7: Hindi Part-of-Speech (POS) Tagging

This directory contains implementations of **Hindi Part-of-Speech (POS) taggers** comparing a feature-rich **Naive Bayes Classifier** against **Hidden Markov Models (HMM)** decoded via three Viterbi variants: **Forward Viterbi**, **Backward Viterbi**, and **Forward-Backward Viterbi**.

---

## File Overview

* **`common.py`**:
  * Shared data loading module parsing both columnar (`word TAG`) and inline (`word/TAG` or `word_TAG`) formats from `Hindi-POS-Data/`.
  * Computes evaluation metrics (per-tag Precision, Recall, F1-score, Support, and overall Accuracy) and exports results to JSON in `results/`.
* **`q1_naive_bayes.py`**:
  * Feature-based Naive Bayes tagger with Add-$\alpha$ smoothing ($\alpha = 0.1$).
  * Feature set extracted per token:
    1. `length`: Binary feature (`MORE` if token length $> 4$, else `LESS`)
    2. `word`: Current word string
    3. `prev_word`: Previous word context ($w_{i-1}$)
    4. `next_word`: Next word context ($w_{i+1}$)
    5. `prev_tag`: Previous tag context ($t_{i-1}$)
* **`hmm_model.py`**:
  * Estimates transition matrices ($P(t_i \mid t_{i-1})$ forward, $P(t_i \mid t_{i+1})$ backward), start/end distributions, and smoothed emission probabilities ($P(w \mid t)$).
  * Out-of-vocabulary (OOV) emission probabilities are modeled using training tokens appearing exactly once (hapax legomena).
  * Includes the general dynamic programming Viterbi trellis decoder.
* **`q2a_forward_viterbi.py`**:
  * Standard left-to-right Forward Viterbi decoding maximizing the joint sequence probability.
* **`q2b_backward_viterbi.py`**:
  * Right-to-left Backward Viterbi decoding scanning from sentence end to start using reversed transitions.
* **`q2c_forward_backward_viterbi.py`**:
  * Bidirectional Viterbi decoder combining forward and backward trellis scores:
    $$\text{score}(i, t) = \text{forward}[i][t] + \text{backward}[i][t] - \ln P(w_i \mid t)$$
* **`compare.py`**:
  * Compares all four approaches, prints comprehensive per-tag F1 summary tables, and exports comparative bar charts (`f1_per_tag_dev.png`, `f1_per_tag_test.png`) to `results/`.

---

## Experimental Results & Comparison

Evaluated on Hindi POS benchmark splits (**13,858 dev tokens**, **14,142 test tokens**):

| Model | Decoding Strategy | Dev Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **`q1_naive_bayes.py`** | Greedy Local Classification (5 Features) | 93.07% | 93.21% |
| **`q2a_forward_viterbi.py`** | **HMM Forward Viterbi (Left $\to$ Right)** | 94.33% | **94.38%** |
| **`q2b_backward_viterbi.py`** | **HMM Backward Viterbi (Right $\to$ Left)** | **94.36%** | 94.32% |
| **`q2c_forward_backward_viterbi.py`** | HMM Forward-Backward Combination | 94.00% | 94.04% |

---

## Key Analysis & Insights

1. **Global Trellis vs. Local Greedy Decisions**:
   * HMM Viterbi models (~94.35%) consistently outperform the Naive Bayes classifier (~93.15%). By optimizing the global tag path across the entire sentence, Viterbi avoids cascading errors that occur when greedy classifiers mispredict a single ambiguous token.

2. **Forward vs. Backward Directional Symmetry**:
   * Forward Viterbi (94.38% test) and Backward Viterbi (94.32% test) achieve virtually identical accuracy. First-order Markov dependencies in Hindi syntax (SOV order, postpositions) are captured equally well when propagating transitions forward or backward.

3. **Per-Tag Strengths**:
   * Postpositions (`PSP`), punctuation (`RD_PUNC`), and verbs (`V_VM`, `V_VAUX`) consistently achieve near-perfect F1-scores ($>0.96$ to $1.00$).
   * Infrequent or open-class categories (e.g., adverbs `RB` and foreign/rare tokens) present greater ambiguity, where HMM transition constraints provide a notable advantage over local word-level features.

---

## How to Run

Execute the models and comparison pipeline in sequence:

```bash
# 1. Run Naive Bayes classifier
python3 q1_naive_bayes.py

# 2. Run Forward Viterbi HMM
python3 q2a_forward_viterbi.py

# 3. Run Backward Viterbi HMM
python3 q2b_backward_viterbi.py

# 4. Run Forward-Backward Viterbi HMM
python3 q2c_forward_backward_viterbi.py

# 5. Generate comparative summary table and visualization charts
python3 compare.py
```
