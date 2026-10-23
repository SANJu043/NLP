import re
import pypdf

PDF_PATH = "./brown_nouns.pdf"
OUT_PATH = "./fst_output1.txt"
INVALID_OUT_PATH = "./invalid_words.txt"

VOWELS_AOU = set("aou")

IRREGULAR_PLURALS = {
    "mice": "mouse", "children": "child", "men": "man", 
    "women": "woman", "teeth": "tooth", "feet": "foot", "geese": "goose"
}

ZERO_PLURALS = {
    "sheep", "deer", "fish", "aircraft", "salmon"
}
SINGULARS_ENDING_IN_S = {
    # Singulars ending in a single 's'
    "bus", "gas", "lens", "canvas", "chaos", "atlas",
    # Words naturally ending in "ies"
    "series", "species", "rabies",
    # False plurals / mass nouns
    "news", "physics", "mathematics", "measles", "gymnastics"
}

def step(state: str, ch: str) -> str:
    """One FST transition: (state, input char) -> next state."""
    if ch == "h":
        if state in ("PENDC", "PENDS"):
            return "SIB"
        return "BORING_C"

    if ch == "c":
        return "PENDC"

    if ch == "s":
        return "PENDS"

    if ch in ("z", "x"):
        return "SIB"

    if ch == "i":
        return "I"

    if ch == "e":
        if state in ("SIB", "PENDS"):
            return "SIBE"
        if state == "I":
            return "IE"
        return "E"

    if ch == "y":
        if state in ("BORING_V", "I", "IE", "E", "SIBE"):
            return "BORING_V"
        return "Y"

    if ch in VOWELS_AOU:
        return "BORING_V"

    return "BORING_C"


def run_fst(stem: str) -> str:
    """Feed every character of `stem` through the FST; return final state."""
    state = "BORING_C"
    for ch in stem:
        state = step(state, ch)
    return state

def analyze(word: str) -> str:
    """'foxes' -> 'fox+N+PL'   |   'foxs' -> '"Invalid Word"' """
    if not word or not all("a" <= c <= "z" for c in word):
        return '"Invalid Word"'

    # --- NEW LEXICON CHECKS ---
    
    if word in IRREGULAR_PLURALS:
        return f"{IRREGULAR_PLURALS[word]}+N+PL"
        
    if word in ZERO_PLURALS:
        return f"{word}+N+SG/PL"

    if word in SINGULARS_ENDING_IN_S:
        return f"{word}+N+SG"

    # --- YOUR ORIGINAL FST LOGIC ---

    if word.endswith("ss"):
        return f"{word}+N+SG"
    
    if not word.endswith("s"):
        return f"{word}+N+SG"

    stem1 = word[:-1]
    final_state = run_fst(stem1)

    if final_state == "IE":
        return f"{stem1[:-2] + 'y'}+N+PL"
    if final_state == "SIBE":
        return f"{stem1[:-1]}+N+PL"
    if final_state in ("SIB", "PENDS"):
        return '"Invalid Word"'
    return f"{stem1}+N+PL"


def generate(root: str, number: str = "SG") -> str:
    """'fox'+PL -> 'foxes'   (the other direction of the SAME transducer)"""
    if number == "SG":
        return root
    state = run_fst(root)
    if state == "Y":
        return root[:-1] + "ies"
    if state in ("SIB", "PENDS"):
        return root + "es"
    return root + "s"


def extract_words(pdf_path: str) -> set:
    reader = pypdf.PdfReader(pdf_path)
    n_pages = len(reader.pages)
    print(f"Opened PDF: {n_pages} pages. Extracting...", flush=True)

    raw_tokens = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        raw_tokens.extend(text.split())
        if i % 50 == 0:
            print(f"  page {i}/{n_pages}", flush=True)

    clean = set()
    for tok in raw_tokens:
        tok = tok.lower()
        if re.fullmatch(r"[a-z]+", tok):
            clean.add(tok)
    return clean


def main():
    words = extract_words(PDF_PATH)
    print(f"Unique clean noun tokens extracted from PDF: {len(words)}")

    valid_lines = []
    invalid_words = []

    for w in sorted(words):
        result = analyze(w)
        valid_lines.append(f"{w} = {result}\n")
        if result == '"Invalid Word"':
            invalid_words.append(w)

    with open(OUT_PATH, "w") as out:
        out.writelines(valid_lines)

    with open(INVALID_OUT_PATH, "w") as out:
        for w in invalid_words:
            out.write(f"{w}\n")

    print(f"Invalid Word count: {len(invalid_words)}")
    print(f"Full output written to {OUT_PATH}")
    print(f"Invalid words written separately to {INVALID_OUT_PATH}")


if __name__ == "__main__":
    main()