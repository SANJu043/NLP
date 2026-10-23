VOWELS = set("aeiou")

def char_class(ch):
    if ch in VOWELS:
        return 'V'
    if ch in ('s', 'x', 'z'):
        return 'SIB'
    if ch == 'c':
        return 'C'
    if ch == 'y':
        return 'Y'
    if ch == 'h':
        return 'H'
    if ch.isalpha():
        return 'O'
    return None


Q0, Q1, Q2, Q3, Q5 = 'q0', 'q1', 'q2', 'q3', 'q5'
START_STATE = Q0
SIBILANT_STATE = Q2
YTRIGGER_STATE = Q5
REJECT = 'REJECT'


DELTA = {
    (Q0, 'V'): Q0, (Q0, 'SIB'): Q2, (Q0, 'C'): Q3, (Q0, 'Y'): Q1, (Q0, 'H'): Q1, (Q0, 'O'): Q1,
    (Q1, 'V'): Q0, (Q1, 'SIB'): Q2, (Q1, 'C'): Q3, (Q1, 'Y'): Q5, (Q1, 'H'): Q1, (Q1, 'O'): Q1,
    (Q2, 'V'): Q0, (Q2, 'SIB'): Q2, (Q2, 'C'): Q3, (Q2, 'Y'): Q5, (Q2, 'H'): Q2, (Q2, 'O'): Q1,
    (Q3, 'V'): Q0, (Q3, 'SIB'): Q2, (Q3, 'C'): Q3, (Q3, 'Y'): Q5, (Q3, 'H'): Q2, (Q3, 'O'): Q1,
    (Q5, 'V'): Q0, (Q5, 'SIB'): Q2, (Q5, 'C'): Q3, (Q5, 'Y'): Q5, (Q5, 'H'): Q1, (Q5, 'O'): Q1,
}

def run_fst(root_string):
    state = START_STATE
    if root_string == "":
        return START_STATE
    for ch in root_string:
        cls = char_class(ch)
        if cls is None:
            return REJECT
        nxt = DELTA.get((state, cls))
        if nxt is None:
            return REJECT
        state = nxt
    return state

def state_is_consonant_like(state):
    return state in (Q1, Q2, Q3, Q5)

def analyze(word, lexicon):
    lw = word.lower()
    if not lw.isalpha() or len(lw) == 0:
        return None  # Invalid Word

    if lw.endswith('ies') and len(lw) > 3:
        cand_y = lw[:-3] + 'y'                 # Y-replacement root, e.g. tries -> try
        cand_plain = lw[:-1]                   # plain S-addition root, e.g. movies -> movie
        y_prefix_state = run_fst(cand_y[:-1])  # state right before the 'y'
        y_structurally_ok = state_is_consonant_like(y_prefix_state)

        if cand_plain in lexicon:
            return (cand_plain, 'N', 'PL')                       # S-addition wins (lexicon-attested)
        if y_structurally_ok and cand_y in lexicon:
            return (cand_y, 'N', 'PL')                           # Y-replacement, lexicon-attested
        if y_structurally_ok:
            return (cand_y, 'N', 'PL')                           # Y-replacement, structural fallback
        return None                                               # Invalid Word

    if lw.endswith('es') and len(lw) > 2:
        cand_ins = lw[:-2]                     # E-insertion root  foxes -> fox
        cand_plain = lw[:-1]                   # plain-S root   houses -> house
        ins_state = run_fst(cand_ins)
        plain_state = run_fst(cand_plain)

        if cand_plain in lexicon:
            return (cand_plain, 'N', 'PL')                         # house/houses, movie/movies
        if cand_ins in lexicon and ins_state == SIBILANT_STATE:
            return (cand_ins, 'N', 'PL')                           # fox/foxes
        if ins_state == SIBILANT_STATE:
            return (cand_ins, 'N', 'PL')
        if plain_state not in (SIBILANT_STATE, YTRIGGER_STATE):
            return (cand_plain, 'N', 'PL')
        return None

    if lw.endswith('s') and len(lw) > 1:
        cand = lw[:-1]
        st = run_fst(cand)
        if st == REJECT:
            return None
        if st in (SIBILANT_STATE, YTRIGGER_STATE):
            if cand in lexicon:
                return None
            return (lw, 'N', 'SG')
        return (cand, 'N', 'PL')

    if run_fst(lw) == REJECT:
        return None
    return (lw, 'N', 'SG')


def load_words(in_path):
    words = []
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            for tok in line.strip().split():
                if tok:
                    words.append(tok)
    return words

def unique_words(words):
    seen = set()
    uniq = []
    for w in words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq

def process_file(in_path, valid_out_path, invalid_out_path):
    all_words = load_words(in_path)
    words = unique_words(all_words)
    lexicon = set(w.lower() for w in words)

    valid_count = 0
    invalid_count = 0
    valid_lines = []
    invalid_lines = []

    for w in words:
        result = analyze(w, lexicon)
        if result is None:
            invalid_lines.append(w)
            invalid_count += 1
        else:
            root, cat, num = result
            valid_lines.append(f"{w} = {root}+{cat}+{num}")
            valid_count += 1

    summary = (
        f"\n----- SUMMARY -----\n"
        f"Total tokens in corpus : {len(all_words)}\n"
        f"Unique words analyzed  : {len(words)}\n"
        f"Valid words            : {valid_count}\n"
        f"Invalid words          : {invalid_count}\n"
    )

    with open(valid_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(valid_lines) + "\n")
        f.write(summary)

    with open(invalid_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(invalid_lines) + "\n")
        f.write(summary)

    return len(all_words), len(words), valid_count, invalid_count


IN_PATH = "brown_nouns.txt"
VALID_OUT_PATH = "valid_words.txt"
INVALID_OUT_PATH = "invalid_words.txt"

if __name__ == "__main__":
    total_tokens, num_unique, valid, invalid = process_file(IN_PATH, VALID_OUT_PATH, INVALID_OUT_PATH)

    print(f"Total tokens in corpus : {total_tokens}")
    print(f"Unique words analyzed  : {num_unique}")
    print(f"Valid words            : {valid}")
    print(f"Invalid words          : {invalid}")
    print(f"\nValid words written to   : {VALID_OUT_PATH}")
    print(f"Invalid words written to : {INVALID_OUT_PATH}")