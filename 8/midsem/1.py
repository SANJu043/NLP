def fst(noun):
    suffixes = ('s', 'x', 'z', 'ch', 'sh')
    if noun.endswith(suffixes):
        singular = noun
        plural = noun + 'es'
    else:
        singular = noun
        plural = noun + 's'
    return f"{singular}+N+SG", f"{plural}+N+PL"

# word = input()
# print(fst(word))
print(fst('cat'))
print(fst('box'))