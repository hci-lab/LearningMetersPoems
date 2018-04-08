import arabic
from itertools import product 
from pyarabic.araby import strip_tashkeel, strip_tatweel
import numpy as np
from numpy import array
import re
import h5py

counter = 0



def update_log_file(exp_name,text,epoch_flage=False):
    
    def update(line,text,epoch_flage):
        if epoch_flage:
                line = line.split("_")[0]+"_"+text
        else:
                line = line = line.split(",")[0]+","+text
        return line
    try:
        lines = open("log.txt").read().split('\n')
        lines = [line if exp_name != line.split(",")[0] else update(line,text,epoch_flage) for line in lines]
        file = open('log.txt','w')
        file.write('\n'.join(lines))
        file.close()
        return True
    except:
        return False


def save_h5(nameOfFile,nameOfDataset,dataVar):
    h5f = h5py.File(nameOfFile, 'w')
    h5f.create_dataset(nameOfDataset, data=dataVar)
    h5f.close()

def separate_token_with_dicrites(token):
    """gets a token(string) with taskeel, and returns a list of strings,
    each string in the list represents each character in the token with its own
tashkeel.
    Args:
        token (str): string represents a word or aya or sura
    Returns:
        [str]: a list contains the token characters with their tashkeel.
    """
    token_without_tatweel = strip_tatweel(token)
    hroof_with_tashkeel = []
    for index,i in enumerate(token):
        if(((token[index] in (arabic.alphabet or arabic.alefat or arabic.hamzat
)) or token[index] is ' ' or  token[index] is "\n" )):
            k = index
            harf_with_taskeel = token[index]
            while((k+1) != len(token) and (token[k+1] in (arabic.tashkeel or 
            arabic.harakat or arabic.shortharakat or arabic.tanwin))):
                harf_with_taskeel =harf_with_taskeel+""+token[k+1]
                k = k + 1
            index = k
            hroof_with_tashkeel.append(harf_with_taskeel)
    return hroof_with_tashkeel


binary = [0,1]
encoding_combination = [list(i) for i in product([0, 1], repeat=8)]

def get_alphabet_tashkeel_combination(tashkeel=arabic.shortharakat):

    '''
        * Creating Letters with (fatha, damma, kasra, sukun) combinations
    '''
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
    arabic_alphabet_tashkeel = [''] + alphabet + arabic_alphabet_tashkeel
        
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
    factoredString = ''
    charsList = separate_token_with_dicrites(string)
    # print(charsList)

    for char in charsList:
        if len(char) < 2:
            factoredString += char
        if len(char) == 2:
            if char[1] in arabic.shortharakat:
                factoredString += char
            elif char[1] ==  arabic.dammatan:
                if char[0] == arabic.teh_marbuta:
                    factoredString += arabic.teh + arabic.damma + \
                       arabic.noon + arabic.sukun
                else:
                    # the letter
                    factoredString += char[0]  + arabic.damma + \
                       arabic.noon + arabic.sukun
            elif char[1] == arabic.kasratan:
                if char[0] == arabic.teh_marbuta: 
                      factoredString += char[0] + arabic.teh + \
                      arabic.kasra + arabic.noon + arabic.sukun
                else:
                    # the letter
                    factoredString += char[0] + arabic.kasra \
                                   + arabic.noon + arabic.sukun
            elif char[1] == arabic.fathatan:
                if char[0] == arabic.alef:
                    factoredString += arabic.noon + arabic.sukun
                elif char[0] == arabic.teh_marbuta:
                    factoredString += arabic.teh + arabic.fatha \
                                   + arabic.noon + arabic.sukun
            elif char[1] == arabic.shadda:
                    factoredString += char[0] + arabic.sukun + char[0]
            
        if len(char) == 3:
            factoredString += char[0] + arabic.sukun + char[0] + char[2]

    return factoredString


'''
print(factor_shadda_tanwin('بيتٌ'))
print(factor_shadda_tanwin('ولدٍ'))
print(factor_shadda_tanwin('ولدَاً'))
print(factor_shadda_tanwin('مدرسةً'))
print(factor_shadda_tanwin('مدرسةٍ'))
print(factor_shadda_tanwin('مدرسةٌ'))
print(factor_shadda_tanwin('شبّ'))
print(factor_shadda_tanwin('كبَّ'))
'''
'''
# Testing
for i in factor_shadda_tanwin('أَشَّدونٌ'):
    print(i)
'''


def string_vectorizer(strng, alphabet=arabic.alphabet):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return  np.array(vector)

def string_with_tashkeel_vectorizer(string, padding_length):
    '''
        return: 8*1 vector representation for each letter in string
    '''

    # 0* change string to list of letters
    '''
        * where tshkeel is not considerd a letter
        > Harakah will no be a single member in list
        > it will be concatinated with its previous letter or not exist
    '''
    # factor shaddah and tanwin
    string = factor_shadda_tanwin(string)

    # harakah is concatinated with previous latter.
    string_clean = separate_token_with_dicrites(string)

    # 1* Building letter and taskell compinations
    arabic_alphabet_tashkeel = lettersTashkeelCombination

    encoding_combination_ = np.array(encoding_combination)

    # 4* getting encoding for each letter from input string
    representation = []
    for x in string_clean:
        index = arabic_alphabet_tashkeel.index(x)

        representation.append(encoding_combination_[index])

    reminder = padding_length - len(representation)
    for i in range(reminder):
        representation.append([0, 0, 0, 0, 0, 0, 0, 0])
    return np.asarray(representation)

# print(len(string_with_tashkeel_vectorizer('أنا')))

def string_with_tashkeel_vectorizer_OneHot(string, padding_length):
    '''
        * encodes each letter in string with ont-hot vector
        * returns a list of one-hot vectors a list of (1*182) vectors
        * letter -> 1*182 vector
    '''
    cleanedString = factor_shadda_tanwin(string)
    charCleanedString = separate_token_with_dicrites(cleanedString)
    vector = [[0 if char != letter else 1 for char in lettersTashkeelCombination]
                  for letter in charCleanedString]

    reminder = padding_length - len(vector)
    for i in range(reminder):
        vector.append([0] * len(lettersTashkeelCombination))

    return np.array(vector) 



def Clean_data(data_frame,max_bayt_len,inplace=False,vectoriz_function=string_with_tashkeel_vectorizer,verse_column_name='Bayt_Text'):
    global counter
    counter = 0
    if not inplace:
        data_frame = data_frame.copy()
    
    def apply_cleaning(s):
        global counter
        try:
            vectoriz_function(s,max_bayt_len)
            print(counter)
            counter+=1
            return s
        except:
            s = solve_conflect(s)
            print(counter)
            counter+=1
            return s

        

    def clean_fun(s):
        if " " in s:
            return " "

        non_remove = arabic.fatha+"|"+arabic.damma+"|"+arabic.kasra+"|"+arabic.sukun
        remove = arabic.dammatan+"|"+arabic.fathatan+"|"+arabic.kasratan
        tashkiel  = re.compile(r'('"("+non_remove+")"+arabic.shadda+")")
        tanwine  =  re.compile(r'('"("+remove+")"+arabic.shadda+")")
        tanwine_  =  re.compile(r'('+arabic.shadda+"("+remove+")"+")")
        spaces_w_tshkieel = re.compile(r'( ('+"|".join(arabic.tashkeel)+'))')
        tanwine_2 = re.compile(r'(('+non_remove+')('+remove+')'')')
        tow_tashkeel = re.compile(r'(('+non_remove+')('+non_remove+')'')')
        tow_tashkeel_ = re.compile(r'(('+remove+')('+remove+')'')')

        lis=list(s)
        for m in tashkiel.finditer(s):
            lis[m.start()] , lis[m.start()+1] = lis[m.start()+1] , lis[m.start()]
        for m in tanwine_.finditer(s):
            del lis[m.start()]
        for m in tanwine.finditer(s):
            del lis[m.start()] 
        for m in tow_tashkeel.finditer(s):
            del lis[m.start()]
        for m in tow_tashkeel_.finditer(s):
            del lis[m.start()]
        for m in spaces_w_tshkieel.finditer(s):
            del lis[m.start()+1]
        for m in tanwine_2.finditer(s):
            del lis[m.start()+1]
        return "".join(lis)


    def solve_conflect(s):
        return "".join([clean_fun(c) for c in separate_token_with_dicrites(s)])
    
    
    our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
    our_alphabets = "".join(our_alphabets)
    data_frame[verse_column_name]    = data_frame[verse_column_name] .apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(r'ّ+', 'ّ', x)).apply(lambda x: x.strip())
    data_frame[verse_column_name] = data_frame[verse_column_name].apply(apply_cleaning)
    return data_frame


'''
encodedLetters = []
for i in lettersTashkeelCombination:
    x = string_with_tashkeel_vectorizer_OneHot(i, 1)
    encodedLetters.append(x)
    
print("------")
print(len(encodedLetters))
print("------")
print(encodedLetters)
print("------")
print("UNIQUE")
print(len(np.unique(encodedLetters, axis=0)))


print("--------------")
osama = 'ألا ليت الشبابُ يعود يوماً'
encoded = string_with_tashkeel_vectorizer_OneHot(osama, 40)
print(encoded.shape)
'''


'''
x = 'ا'
print(string_with_tashkeel_vectorizer_OneHot(x, 2).shape)
print(string_with_tashkeel_vectorizer_OneHot(x, 2))
'''

'''
# Exhaustive test for 8bit encoding
all_traing = []
for x in get_alphabet_tashkeel_combination():
    all_traing.append(x)
    print(string_with_tashkeel_vectorizer(x))
print(len(all_traing))
'''



'''
[ -1------------ ] -> (1, 183)
[ -------------- ] -> (2, 183)
[ -------------- ] -> (3, 183)
[ -------------- ]
[ ----------1--- ]
[ -------------- ]
[ ---1---------- ]
[ --------1----- ]
[ -------------- ]
[ 0000000 .. 0000]
[ 0000000 .. 0000]
[ 0000000 .. 0000]
[ 0000000 .. 0000]
[ 0000000 .. 0000]
[ 0000000 .. 0000]
[ 0000000 .. 0000] -> (190, 183)
'''
