# Q2a - HMM with standard (left-to-right) Viterbi
from common import load_all, evaluate_and_save
from hmm_model import HMM, viterbi

data = load_all()
hmm = HMM(data["train"])


def predict(words):
    tags, _ = viterbi(words, hmm, hmm.start_f, hmm.trans_f)
    return tags


if __name__ == "__main__":
    evaluate_and_save("hmm_forward_viterbi", predict, data)
