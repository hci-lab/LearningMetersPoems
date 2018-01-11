#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import array
from numpy import argmax

# =============================================================================

# =============================================================================
np.random.seed(7)
english_alphabet = 'abcdefghijklmnopqrstuvwxyz '
numberOfUniqueChars = len(english_alphabet)
# =============================================================================

# vectorize fanction 
# =============================================================================
def string_vectorizer(strng, alphabet=arabic_alphabet):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return array(vector)
# =============================================================================


# read data
# =============================================================================
sample_english_poetry = pd.read_csv("./data/All_Data.csv", sep = ",")
max_Bayt_length =  sample_english_poetry.verse.map(len).max()
# =============================================================================


# =============================================================================
Verse_Text_Encoded = sample_arabic_poetry['Bayt_Text'].apply(string_vectorizer)


######******************save oure encoding ***************************



# =============================================================================
#one hot encoding for classes
# =============================================================================
Bayt_Bahr = sample_arabic_poetry['Category']
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Bayt_Bahr)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Bayt_Bahr_encoded = onehot_encoder.fit_transform(integer_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(Bayt_Bahr_encoded[1, :])])
print(inverted)
# =============================================================================
 
