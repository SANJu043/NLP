data = [
    ([1, 1, 1], 'Spam'),
    ([1, 1, 0], 'Spam'),
    ([1, 0, 1], 'Spam'),
    ([0, 1, 1], 'Spam'),
    ([0, 0, 1], 'NotSpam'),
    ([0, 1, 0], 'NotSpam'),
    ([1, 0, 0], 'NotSpam')
]

total = len(data)
spam_count = sum(1 for d, c in data if c == 'Spam')
not_spam_count = total - spam_count
features_E8 = [0, 0, 0]

def calc_prob(class_label):
    class_data = [d for d, c in data if c == class_label]
    c_len = len(class_data)
    prob = (spam_count if class_label == 'Spam' else not_spam_count) / total
    for i, val in enumerate(features_E8):
        match = sum(1 for d in class_data if d[i] == val)
        prob *= (match / c_len)
    return prob

p_e8_spam = calc_prob('Spam')
p_e8_notspam = calc_prob('NotSpam')
prediction = 'Spam' if p_e8_spam > p_e8_notspam else 'NotSpam'

print(f"Spam Prob: {p_e8_spam}")
print(f"NotSpam Prob: {p_e8_notspam}")
print(f"Prediction for E8: {prediction}")