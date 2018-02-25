import pandas as pd
import re
import sys
import helpers
from pyarabic.araby import strip_tatweel
sys.path.append('.') # path of arabic file
import arabic 
vectoriz_function = helpers.string_with_tashkeel_vectorizer
counter = 0

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


def apply_cleaning(s):
    global counter
    try:
        global vectoriz_function
        vectoriz_function(s)
        print(counter)
        counter+=1
        return s
    except:
        s = solve_conflect(s)
        print(counter)
        counter+=1
        return s
        #helpers.string_with_tashkeel_vectorizer(s)
        

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


def Clean_data(data_frame,verse_column_name='البيت'):
    our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
    our_alphabets = "".join(our_alphabets)
    data_frame[verse_column_name]    = data_frame[verse_column_name] .apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(r'ّ+', 'ّ', x)).apply(lambda x: x.strip())
    data_frame[verse_column_name] = data_frame[verse_column_name].apply(apply_cleaning)
    return data_frame


################### how to use #############
# Example::
# 
# un_clean_data = pd.read_csv('data.csv') 
# clean_data = Clean_data(un_clean_data)
#
##########################################3

    
