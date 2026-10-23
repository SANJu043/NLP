corpus = [
    "low lower low lower lowest",
    "low lowest low lowest",
    "low new",
    "new new newer newer",
    "new"
]

words = " ".join(corpus).split()
vocab = [" ".join(list(w)) + " </w>" for w in words]

def wordpiece_tokenize(word):
    chars = " ".join(list(word)) + " </w>"
    return chars

print(wordpiece_tokenize("newest"))