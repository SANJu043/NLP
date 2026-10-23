# Text Segmentation Benchmarks: Dynamic Programming vs. Greedy Approaches

This repository contains Python implementations and evaluations of different word segmentation algorithms used to split unsegmented character strings (e.g., `itthatthecitytakestepstothisproblem`) into valid sequences of words.

The project evaluates probabilistic sequence modeling via **Dynamic Programming (DP)** alongside baseline and ensemble **Greedy Longest-Match** heuristics across 1,000 test cases using unigram word statistics from `text_segmentation_dataset.json`.

---

## File Overview & Implementation Details

### 1. `dp_segment.py`
* **Algorithm**: Dynamic Programming with Max Log-Probability (Unigram Language Model).
* **How it works**:
  * Loads word frequencies from the dataset and computes log probabilities $P(w) = \\ln\\left(\\frac{\\text{count}(w)}{\\text{total\\_count}}\\right)$.
  * Employs smoothing for out-of-vocabulary (OOV) tokens with a penalty function based on word length.
  * Solves subproblems iteratively: `best_score[i]` stores the maximum cumulative log probability to segment the prefix `text[:i]`.
  * Reconstructs the optimal sequence of words via backtracking (`best_split`).
* **Key Metric**: Evaluates Levenshtein edit distance at both the word level and character level along with exact-match accuracy.

### 2. `greedy_segment.py`
* **Algorithm**: Baseline Greedy Longest-Match Segmentation.
* **How it works**:
  * **Left to Right (`greedy_segment_ltor`)**: Scans from left to right, matching the longest available substring found in the vocabulary starting at index `i`.
  * **Right to Left (`greedy_segment_rtol`)**: Scans from right to left, matching the longest available substring ending at index `i`.
  * If no matching word is found, it falls back to single characters.

### 3. `exp.py`
* **Algorithm**: Extended Greedy Experiments & Bidirectional Selection.
* **How it works**:
  * Evaluates **Left-to-Right**, **Right-to-Left**, and an ensemble **"Best of Both" (Bidirectional)** strategy.
  * **Bidirectional Logic (`bidirectional_greedy`)**:
    1. Runs both L-to-R and R-to-L passes on the input.
    2. Counts unknown single-character tokens in each candidate segmentation.
    3. Prefers the segmentation with fewer unknown tokens.
    4. Breaks ties by picking the candidate with fewer overall words (favoring longer, valid words).

---

## Experimental Results & Comparison

Evaluated on **1,000 test cases** using `text_segmentation_dataset.json`:

| File / Strategy | Algorithm / Approach | Exact-Match Accuracy | Avg Word Edit Distance | Avg Char Edit Distance |
| :--- | :--- | :---: | :---: | :---: |
| **`dp_segment.py`** | **Dynamic Programming (Max Log-Prob)** | **98.20%** | **0.0370** | **0.0280** |
| `exp.py` | Greedy: Best of Both (Bidirectional) | 92.80% | 0.1750 | 0.1700 |
| `greedy_segment.py` / `exp.py` | Greedy: Right To Left | 88.40% | 0.3840 | 0.3770 |
| `greedy_segment.py` / `exp.py` | Greedy: Left To Right | 69.10% | 1.2600 | 1.2900 |

---

## Key Analysis & Insights

1. **Dynamic Programming vs. Greedy**:
   * **DP (98.20%)** significantly outperforms all greedy variants. By evaluating all possible segmentations and selecting the sequence that maximizes global log-probability, DP avoids local sub-optimal traps.
   * Standard Left-to-Right greedy matching often over-segments compound or adjacent words (e.g., matching `takes` instead of `take` + `steps`, causing remaining letters `t e p s` to fragment into single characters).

2. **Asymmetry in Directional Greedy Matching**:
   * **Right-to-Left (88.40%)** performs noticeably better than **Left-to-Right (69.10%)**. English language morphological patterns and trailing suffixes make right-to-left scanning less susceptible to premature prefix matches in this dataset.

3. **Bidirectional Heuristic Improvement**:
   * Combining both directions into **Best of Both** boosts greedy accuracy to **92.80%**, demonstrating that candidate selection based on minimum unknown tokens and word count effectively recovers from many single-direction failure modes.

---

## How to Run

Execute each script via the terminal:

```bash
python3 dp_segment.py
python3 greedy_segment.py
python3 exp.py