# Assignment 3: English Word Recognition DFA & Morphological Analysis FST

This directory contains implementations of a **Deterministic Finite Automaton (DFA)** for validating English word syntax and a **Finite State Transducer (FST)** for English noun morphological parsing and pluralization based on the Brown Corpus.

---

## File Overview

* **`DFA.py`**:
  * Implements a 3-state DFA ($q_0$: start, $q_1$: accept, $q_D$: trap) that accepts valid simplified English words (`[a-z]+`: begins with a lowercase letter, followed only by lowercase letters, no spaces, digits, punctuation, or uppercase letters).
  * Includes automated state diagram generation using `visual-automata` (`VisualDFA`).

* **`FST1.py`**:
  * Rule-based and lexicon-aware Finite State Transducer for morphological analysis (mapping surface plural words to `root+N+PL`, singulars to `root+N+SG`, and ill-formed words to `"Invalid Word"`).
  * Handles irregular plurals (`mice` $\to$ `mouse+N+PL`), zero-plurals (`sheep` $\to$ `sheep+N+SG/PL`), and singulars ending in *-s* (`bus`, `gas`, `series`, `physics`).
  * Features both **surface $\to$ analysis** (`analyze()`) and **root $\to$ inflected surface** (`generate()`) transductions.

* **`FST2.py`**:
  * Formal state-transition implementation of the FST over character classes: Vowels (`V`), Sibilants (`SIB`: *s, x, z*), Consonants (`C`), `Y`, and `H`.
  * Evaluates tokens from `brown_nouns.txt`, logging valid parses to `valid_words.txt` and ill-formed words to `invalid_words.txt`.

* **`fst_transition_table.csv`**:
  * Explicit state transition matrix mapping states (`Q0`, `Q_VOWEL`, `Q_C`, `Q_S`, `Q_SIB`, `Q_SIBE`, `Q_IE`, `Q_DEAD`) against input letters `a`–`z`, alongside terminal morphological action mappings.

* **`NLP_Assignment_2_DFA_FST_LOCAL.ipynb`**:
  * Interactive notebook with test cases, visualizations (`Digraph.gv`), and step-by-step transducer evaluation.

---

## Morphophonological Rules Implemented

| Rule Name | Linguistic Condition | Surface Transformation | Example |
| :--- | :--- | :--- | :--- |
| **E-insertion** | Root ends in sibilants (*-s, -z, -x, -ch, -sh*) | Insert `e` before suffix `-s` | `fox` $\to$ `foxes`, `watch` $\to$ `watches` |
| **Y-replacement** | Root ends in consonant + `-y` | Change `-y` to `-ie-` before `-s` | `try` $\to$ `tries`, `baby` $\to$ `babies` |
| **S-addition** | Default / regular stem ending | Append `-s` | `bag` $\to$ `bags`, `cat` $\to$ `cats` |
| **Invalid Rejection**| Sibilant/Y-stem directly followed by `-s` without rule | Output `"Invalid Word"` | `foxs` $\to$ `"Invalid Word"` |

---

## Corpus Statistics (`brown_nouns.txt`)

* **Total tokens analyzed**: 285,026
* **Unique noun vocabulary**: 28,145
* **Valid parses**: 27,882
* **Rejected / invalid parses**: 263

---

## How to Run

```bash
# Run the word recognition DFA
python3 DFA.py

# Run FST morphological analysis on Brown nouns (PDF extraction pipeline)
python3 FST1.py

# Run FST with character classes and transition table on brown_nouns.txt
python3 FST2.py
```
