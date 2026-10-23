# Q2b - HMM with backward (right-to-left) Viterbi
# Read the sentence from the end; use P(tag_i | tag_i+1) as transition probability.
from common import load_all, evaluate_and_save
from hmm_model import HMM, viterbi

data = load_all()
hmm = HMM(data["train"])


def predict(words):
    tags, _ = viterbi(words[::-1], hmm, hmm.start_b, hmm.trans_b)   # reversed sentence
    return tags[::-1]                                              # reverse tags back


if __name__ == "__main__":
    evaluate_and_save("hmm_backward_viterbi", predict, data)
