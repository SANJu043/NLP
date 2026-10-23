"""1. Construct a deterministic finite automaton (DFA) that recognizes valid simplified English
    words. An English word, for this problem, is defined as a string that:
    ● Starts with an lowercase letter
    ● Followed by zero or more lowercase letters
    That is, the string must:
    ● Not start with a digit or punctuation
    ● Not contain uppercase letters after the first character
    ● Not contain spaces, digits, or special characters
    Output Format:
    If a string is accepted, output Accepted else Not Accepted.
    ● Accepted: cat, dog, a, zebra
    Not Accepted: dog1, 1dog, DogHouse, Dog_house, cats (starts with a space)
    For visualizing the DFA, you can use https://pypi.org/project/visual-automata/ or
    https://github.com/rohaquinlop/automathon"""
    
# import re

# # Building the pattern
# pattern = re.compile(r"[a-z]+")
# print(re.match(pattern, input("Enter the String: ")))

def dfa_lowercase_word(word):
    state = 'q0'
    
    for char in word:
        if  state == 'q0':
            if 'a' <= char <= 'z':
                state = 'q1'
            else:
                state = 'qd'
        elif state == 'q1':
            if 'a' <= char <= 'z':
                continue
            else:
                state = 'qd'
        elif state == 'qd':
            break
    
    if state == 'q1':
        return "Accepted"
    else:
        return "Not Accepted"

import string
from visual_automata.fa.dfa import VisualDFA

def build_visual_dfa():
    lowercase_letters = set(string.ascii_lowercase)
    
    other_chars = set(string.ascii_uppercase + string.digits + string.punctuation + " ")
    
    all_symbols = lowercase_letters.union(other_chars)

    transitions = {
        "q0": {},
        "q1": {},
        "qD": {}
    }

    for char in all_symbols:
        if char in lowercase_letters:
            transitions["q0"][char] = "q1"
            transitions["q1"][char] = "q1"
        else:
            transitions["q0"][char] = "qD"
            transitions["q1"][char] = "qD"
            
        transitions["qD"][char] = "qD"

    dfa = VisualDFA(
        states={"q0", "q1", "qD"},
        input_symbols=all_symbols,
        transitions=transitions,
        initial_state="q0",
        final_states={"q1"}
    )
    
    return dfa

dfa = build_visual_dfa()

print("Checking 'cat':", dfa.input_check("cat"))
print("Checking 'dog1':", dfa.input_check("dog1"))
print("Checking 'DogHouse':", dfa.input_check("DogHouse"))

# dfa.show_diagram(view=True)