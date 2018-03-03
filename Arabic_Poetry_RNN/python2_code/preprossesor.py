#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:52:27 2018

@author: Mostafa Alaa
"""
import arabic
import pyarabic.araby as araby
import helpers
from helpers import string_with_tashkeel_vectorizer,string_vectorizer
import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#drop_cols = old_data_col

def get_input_encoded_date(input_data_path,drop_cols,with_tashkeel_flag):
    input_arabic_poetry_dataset = pd.read_csv(input_data_path, sep = ",")
    input_arabic_poetry_dataset.drop(input_arabic_poetry_dataset.columns[drop_cols], axis=1,inplace=True)
    input_arabic_poetry_dataset.columns = [u'Bayt_Text', u'Category']
    our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
    our_alphabets = "".join(our_alphabets)
    input_arabic_poetry_dataset[u'Bayt_Text'] = input_arabic_poetry_dataset[u'Bayt_Text'].apply(lambda x: re.sub(r'(^'+our_alphabets+')','',x)).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(u'ّ+', u'ّ', x)).apply(lambda x: x.strip())
    
    if(with_tashkeel_flag == 0):        
        input_arabic_poetry_dataset[u'Bayt_Text'] = input_arabic_poetry_dataset[u'Bayt_Text'].apply(araby.strip_tashkeel).apply(helpers.strip_tatweel)
    
    max_Bayt_length =  input_arabic_poetry_dataset.Bayt_Text.map(len).max()
    Bayt_Text_Encoded = input_arabic_poetry_dataset[u'Bayt_Text'].apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, max_Bayt_length))
    print("Input Data Bayt_Text encoded done.")
    Bayt_Text_Encoded_Stacked = np.stack(Bayt_Text_Encoded,axis = 0)    
    
    numbber_of_bohor = input_arabic_poetry_dataset[u'Category'].unique().size
    Bayt_Bahr_encoded,label_encoder_output = get_classes_encoded_date(input_arabic_poetry_dataset['Category'])
    
    classes_freq = input_arabic_poetry_dataset[u'Category'].value_counts().reset_index()
    classes_freq.columns =['Bohor','Cnt']
    
    
    #return  Bayt_Text_Encoded_Stacked, Bayt_Bahr_encoded, numbber_of_bohor, label_encoder_output, classes_freq

    return  Bayt_Text_Encoded_Stacked, Bayt_Bahr_encoded, max_Bayt_length, label_encoder_output, classes_freq
        

def get_classes_encoded_date(input_category_datasets):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(input_category_datasets)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Bayt_Bahr_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("Input Data Category encoded done.")
    return Bayt_Bahr_encoded,label_encoder


def print_model(hist):
    # list all data in history
    print(hist.history.keys())
    # summarize history for accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# =============================================================================
#     # invert first example
#     inverted = label_encoder.inverse_transform([argmax(Bayt_Bahr_encoded[1, :])])
#     print(inverted)
# =============================================================================

# =================================Padding=====================================

#X_train_padded = sequence.pad_sequences(X_train, maxlen=max_Bayt_length)
#X_test_padded = sequence.pad_sequences(X_test, maxlen=max_Bayt_length)

#print("Padding done.")
# =============================================================================
