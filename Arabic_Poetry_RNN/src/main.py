#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
import pandas as pd
import tensorflow as tf
import os
import pyarabic.araby as araby
import pyarabic.number as number
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import array
from numpy import argmax
# =============================================================================

# =============================================================================
np.random.seed(7)
os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/GP-Ripo-master/Arabic_Poetry_RNN/")
arabic_alphabet = [' ','ب','ة' ,'ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي','ء','آ','أ','ؤ','ئ','\n','ا']
numberOfUniqueChars = len(arabic_alphabet)
# =============================================================================

# =============================================================================
def string_vectorizer(strng, alphabet=arabic_alphabet):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return array(vector)
# =============================================================================

# =============================================================================
sample_arabic_poetry = pd.read_csv("./data/All_Data.csv", sep = ",")
cols = [1,2,4]
sample_arabic_poetry.drop(sample_arabic_poetry.columns[cols], axis=1,inplace=True)
sample_arabic_poetry.columns = ['Bayt_Text', 'Category']
sample_arabic_poetry['Bayt_Text'] = sample_arabic_poetry['Bayt_Text'].apply(araby.strip_tashkeel).apply(araby.strip_tatweel)
max_Bayt_length =  sample_arabic_poetry.Bayt_Text.map(len).max()    
# =============================================================================


# =============================================================================
Bayt_Text_Encoded = sample_arabic_poetry['Bayt_Text'].apply(string_vectorizer)

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


# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(Bayt_Text_Encoded, Bayt_Bahr_encoded, test_size=0.2, random_state=0)
#default padding need to check the paramters details
X_train_padded = sequence.pad_sequences(X_train, maxlen=max_Bayt_length)
X_test_padded = sequence.pad_sequences(X_test, maxlen=max_Bayt_length)
# =============================================================================

# =============================================================================
# Initialising the RNN
model = Sequential()

model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = ( 82, 35)))
# Adding the input layer and the LSTM layer

# Adding the output layer
model.add(Dense(units = 11))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model.summary())

# Fitting the RNN to the Training set
model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test_padded, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores*100))
# =============================================================================



# =============================================================================
# #This allows for characters to be represented by numbers
# CharsForids = {char:Id for Id, char in enumerate(chars)}
# #This is the opposite to the above
# idsForChars = {Id:char for Id, char in enumerate(chars)}
# #How many timesteps e.g how many characters we want to process in one go
# numberOfCharsToLearn = 1
# 
# #letters = "العربية"
# # integer encode input data
# integer_encoded = [CharsForids[char] for char in letters]
# print(integer_encoded)
# 
# # one hot encode
# onehot_encoded = list()
# for value in integer_encoded:
# 	letter = [0 for _ in range(len(chars))]
# 	letter[value] = 1
# 	onehot_encoded.append(letter)
# print(onehot_encoded)
# # invert encoding
# inverted = int_to_char[argmax(onehot_encoded[0])]
# print(inverted)
# sss = string_vectorizer(letters)
# =============================================================================
