import arabic


def get_alphabet_tashkeel_combination(tashkeel=arabic.shortharakat):

    arabic_alphabet_tashkeel = []
    for haraka in arabic.shortharakat:
        for letter in arabic.alphabet:
                arabic_alphabet_tashkeel.append(letter + haraka)

    '''
        * Adding alphabets without tashkeel in front of 
          the letters/tashkeel combination
        * [] => alphabet without taskell then alphabet with fatha, ...
    '''
    alphabet = [] + arabic.alphabet
    alphabet += ' '
    alphabet += '\n'
    arabic_alphabet_tashkeel = alphabet + arabic_alphabet_tashkeel
        
    return arabic_alphabet_tashkeel

'''
# Testings
for i in get_alphabet_tashkeel_combination():
    print(i)

for i  in arabic.alphabet:
    print(i)
print(len(arabic.alphabet))
print(arabic.alphabet)
'''
