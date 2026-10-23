# Q2c - HMM with forward-backward Viterbi
# Run forward Viterbi (left->right) and backward Viterbi (right->left).
# For every word and tag:  combined = forward_score + backward_score - emission
# (the emission is counted in both passes, so we remove it once). Pick the best tag per word.
from common import load_all, evaluate_and_save
from hmm_model import HMM, viterbi

data = load_all()
hmm = HMM(data["train"])


def predict(words):
    n = len(words)
    _, fwd = viterbi(words, hmm, hmm.start_f, hmm.trans_f)
    _, bwd = viterbi(words[::-1], hmm, hmm.start_b, hmm.trans_b)
    bwd = bwd[::-1]                       # so that bwd[i] belongs to word i
    result = []
    for i in range(n):
        best = max(hmm.tags, key=lambda t: fwd[i][t] + bwd[i][t] - hmm.emission(words[i], t))
        result.append(best)
    return result


if __name__ == "__main__":
    evaluate_and_save("hmm_forward_backward_viterbi", predict, data)
