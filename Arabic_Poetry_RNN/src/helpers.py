import arabic
from itertools import product 

binary = [0,1]
encoding_combination = [list(i) for i in product([0, 1], repeat=8)]

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

# Exporting
lettersTashkeelCombination = get_alphabet_tashkeel_combination()


def letter_of_encoding(encodingList):
    '''
    encodingList([]): 8 bit encoding.
    return (a, b):
        a -> can be 0-4 (0 for no tahskeel, 1 fatha, 2 damma, 3 kasra, 4 sukun)
        b -> is the charachter 

    '''
    index  = encoding_combination.index(encodingList)
    letter = lettersTashkeelCombination[index]

    # if it has tashkeel
    if len(letter) == 1:
        return letter, 0

    elif len(letter) == 2:
        #index starts at 0.
        return letter[0], arabic.shortharakat.index(letter[1])+1 
        
'''
# Testings
for i in range (182):
    print(letter_of_encoding(encoding_combination[i]))
'''

def factor_shadda_tanwin(string):
    '''
    * factors shadda to letter with sukun and letter
    * factors tanwin to ?????????
    # Some redundancy is simpler. :"D
    '''
    factoredString = []
    i = 0
    for x in string:
        # if it's shadda factor it to sukun and previous letter
        if x == arabic.shadda:
            factoredString.append(arabic.sukun)
            factoredString.append(string[i-1])
            i += 1
        
        # if it's not shadda add it to the new string.
        elif x == arabic.kasratan:
            factoredString.append(arabic.kasra)
            factoredString.append(arabic.noon)
            factoredString.append(arabic.sukun)
            i += 1
        elif x == arabic.dammatan:
            factoredString.append(arabic.damma)
            factoredString.append(arabic.noon)
            factoredString.append(arabic.sukun)
            i += 1
        else:
            factoredString.append(x)
            i += 1

    return factoredString

'''
# Testing
for i in factor_shadda_tanwin('أَشَّدونٌ'):
    print(i)
'''
