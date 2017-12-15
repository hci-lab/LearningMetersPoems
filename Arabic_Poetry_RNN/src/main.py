#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import pyarabic.araby as araby
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import array
from numpy import argmax

# =============================================================================

# =============================================================================
#run on CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 
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

#Params
checkpoint_best_only = 1

read_from_checkpoints = 0
# =========================With three hidden Layer================================
# create model
model = Sequential()

# Adding the input layer and the LSTM layer
model.add(LSTM(units = 500, input_shape = ( 82, 35)))

# Adding the one hidden layer
model.add(Dense(units = 500,activation = 'relu'))

model.add(Dense(units = 500,activation = 'relu'))

model.add(Dense(units = 500,activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 11,activation = 'softmax'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics = ['accuracy'] )

print(model.summary())
#https://machinelearningmastery.com/check-point-deep-learning-models-keras/
if (read_from_checkpoints == 1 and  checkpoint_best_only == 1 ):
    print('Model Will read from Check points with Best Result only')
    model.load_weights("weights.best.hdf5")

if (read_from_checkpoints == 1 and  checkpoint_best_only == 0 ):
    print('Model Will read from Check points from the checkpoints files')
    model.load_weights("weights.best.hdf5")

if(checkpoint_best_only == 1 and read_from_checkpoints == 0):
# checkpoint
    print('Model Will Train and save the best results only')
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

if(checkpoint_best_only == 0 and read_from_checkpoints == 0):
    print('Model Will Train and save the all wights results only')
    filepath="weights-improvement-exp3-{epoch:20}-{three_layer}-{units:500}.hdf5"
    #filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

# Fitting the RNN to the Training set
model.fit(X_train_padded, y_train, validation_split = 0.2, epochs=20, batch_size=32, callbacks=callbacks_list, verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test_padded, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# =========================With one hidden Layer================================

# =============================================================================
# model = Sequential()
# 
# model.add(LSTM(units = 100, input_shape = ( 82, 35)))
# # Adding the input layer and the LSTM layer
# 
# # Adding the output layer
# model.add(Dense(units = 100,activation = 'relu'))
# 
# # Adding the output layer
# model.add(Dense(units = 11,activation = 'softmax'))
# 
# # Compiling the RNN
# model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics = ['accuracy'] )
# print(model.summary())
# 
# # Fitting the RNN to the Training set
# model.fit(X_train_padded, y_train, validation_split = 0.1, epochs=20, batch_size=32)
# 
# # Final evaluation of the model
# scores = model.evaluate(X_test_padded, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# 
# 
# =============================================================================
# =========================Without any hidden Layer========================================
# =============================================================================
# # Initialising the RNN
# model = Sequential()
# 
# model.add(LSTM(units = 100, input_shape = ( 82,
#                                             35)))
# # Adding the input layer and the LSTM layer
# 
# # Adding the output layer
# model.add(Dense(units = 11,activation = 'softmax'))
# 
# # Compiling the RNN
# model.compile(optimizer = 'adam', loss='categorical_crossentropy' )
# print(model.summary())
# 
# # Fitting the RNN to the Training set
# model.fit(X_train_padded, y_train, validation_split = 0.1, epochs=20, batch_size=64)
# 
# # Final evaluation of the model
# scores = model.evaluate(X_test_padded, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores*100))
# 
# =============================================================================
