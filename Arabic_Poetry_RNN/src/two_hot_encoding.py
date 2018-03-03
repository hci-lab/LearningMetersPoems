# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 01:37:00 2018

@author: ali

we will encode each character into a vector using 2-of-k encoding 
(i.e. all zero except for douple ones,
the first one at the index of the character in the sorted alphabet, 
the second one at the index of the Diacritic in the Diacritics )

This encoding with work for any language have symbols on the alphabet, 
these symbols should have it'd one unicode (stand-alone symbols) 


How does this work ?
1- select the alphabet
2- select the Diacritics 
3- select the max length of the text (just castomize this for uor task)
4-apply it on your pandas series after preprocessing (factor shadda , tanween and so on)
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import time
import h5py
from sklearn.preprocessing import OneHotEncoder,  LabelEncoder
import arabic
from pyarabic.araby import strip_tashkeel, strip_tatweel


#------------------------Helper function -----------------------------
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


def encode_one_hot(values):


    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    return onehot_encoded


#-------------------------Program Constants---------------------------
dataPath="C:\\Users\\ali\\Documents\\GitHub\\ArabicPoetry-1\\Arabic_Poetry_RNN\\data\\All_Data.csv"

#One Hot Encoding Dict For All Alphabet
#generate one hot for alphabet and save it in dictionary +" "
    #alphabet+space
alphabet=[" "]+sorted(list(arabic.alphabet))
alphabetEncoded=encode_one_hot(alphabet)
print(alphabetEncoded)
#save One hot encoding in a dict
i=0
oneHotAlphapetDict = {}
for hot in alphabetEncoded:
    oneHotAlphapetDict.update({alphabet[i]: hot})
    i=i+1


#one hot Encoding Dict For All diacritics

#generate one hot for diacritics and save it in dictionary
# noTashkeel "$"-> 1, fatha ->2 , Damma->3 , kasra ->4, skon-> 5
diacritics = ["$"]+ list(arabic.shortharakat)
#diacritics = list(arabic.shortharakat)
diacriticsEncoded=encode_one_hot(diacritics)
oneHotDiacriticsDict = {}
i=0
for hot in diacriticsEncoded:
    oneHotDiacriticsDict.update({diacritics[i]: hot})
    i=i+1

maxLength=83

# -------------------Importing the Data set----------------------------------
dataset = pd.read_csv(dataPath)# encoding = "ISO-8859-1", encoding='utf_8'
dataset.head()
#dataset.iloc[:,3].value_counts()





#text pass only with these diacritics , so separate others like shadah

def two_hot_encoding(string, maxLength, asTwoHotPadding=False):
    # padding text with max length
    string = string + ' ' * (maxLength)

    #   separate each char with it's tashkel
    separatedAlphabetWithDicrites= separate_token_with_dicrites(string)
    print(separatedAlphabetWithDicrites)
    lineEncoded=[]


    #loop on it and separate each index to to char Ex: alef+fatha
    #then replace one-hot encoding of alef + one-hot encoding of fatha
    if not asTwoHotPadding:
        for indx in separatedAlphabetWithDicrites:
            if len(indx)==1:
                twoHotVec=list(oneHotAlphapetDict.get(
                    indx))+list(oneHotDiacriticsDict.get("$"))#[1,0,0,0,0]
                          #+oneHotDiacriticsDict.get( "$")
            else #if len(indx)==2::
                twoHotVec=list(oneHotAlphapetDict.get(indx[0]))+list(oneHotDiacriticsDict.get(indx[1]))
            lineEncoded.append(twoHotVec)
    #incase you striped tashkel and want two encode each char as two hot vector
    else:
        for indx in separatedAlphabetWithDicrites:
            twoHotVec = list(oneHotAlphapetDict.get(indx)) + list(oneHotDiacriticsDict.get("$"))
            lineEncoded.append(twoHotVec)

    # it will return N*P , N is the max length ,
    # and P equals len(alphabet)+ len(diacritics)+ 1
    return lineEncoded




#apply it on each payt "PREPROCESS EL-BYT BEFORE PASS IT"
'''
twoHotMatrix=[]
for verse in coulmn:
    #preprocess your verse before encoding 
    
    lineEncoded=two_hot_encoding(verse)
    twoHotMatrix.append(lineEncoded)
'''



#This part for test
#print(dataset.head())
print("--------------")
byt="وَكانَ في الباطِلِ اِبتِهالي وَاِبتَهَلَ الدَهرُ في أَذاتي"
print(byt[0])
print(two_hot_encoding(byt))
x=separate_token_with_dicrites(byt)[0]
print(len(arabic.alphabet))
#print(list(oneHotAlphapetDict.get(x[0]))+[1, 0])
#print(oneHotDiacriticsDict.get("$"))
'''
print(oneHotAlphapetDict.get("ب"))
print(oneHotAlphapetDict.get("ت"))
print(oneHotAlphapetDict.get("ث"))
print(oneHotAlphapetDict.get("ج"))
'''
print(alphabet)
print(two_hot_encoding(byt))
#print(len(byt))