# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:58:10 2018

@author: ali
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import seaborn as ses
import string
import re
from IPython.display import clear_output
import time
import h5py
from sklearn.preprocessing import OneHotEncoder,  LabelEncoder
from keras.utils import to_categorical
import math




#==================================Clean data==================================
# dataset = pd.read_csv('../data/english_dataset.csv', encoding = "utf-8",index_col=0)
# dataset.columns = ['Verse','Meter']
# dataset['Verse'] = dataset['Verse'].map(lambda x: x.lower())
# dataset['Meter'] = dataset['Meter'].map(lambda x: x.lower())
# our_alphabets="".join(list(string.ascii_lowercase)+[" ","\'"])
# dataset['Verse']=dataset['Verse'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(
#                                           lambda x: re.sub(r'  *'," ",x))
# dataset['Meter']=dataset['Meter'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(
#                                           lambda x: re.sub(r'  *'," ",x))
# dataset['Verse']=dataset['Verse'].apply(lambda x: x.strip())
# dataset['Meter']=dataset['Meter'].apply(lambda x: x.strip())
# dataset.to_csv('../data/english_dataset.csv', encoding = "utf-8")



#==================================Program Constants ==========================
english_alphapets=list(string.ascii_lowercase)
alphapet=english_alphapets+[' ', '\''] #Add space and apostrophe to our alphabets
len_of_alphapet=len(alphapet)

#Decimal Encoding Dict For All Alphabet 
decimal_alphapet_dict={}
i=1
for char in alphapet:
    decimal_alphapet_dict.update({char: i})
    i=i+1
#Binary Encoding Dict For All Alphabet 
binary_alphapet_dict={}
for key, value in decimal_alphapet_dict.items():
    binary_string=[int(i) for i in np.binary_repr(value,  width=5)]
    binary_alphapet_dict.update({key:binary_string})



#==================================Importing Data set==========================
dataset = pd.read_csv('../merged_data.csv', encoding='utf-8' ,index_col=0)
#data= shuffle(dataset)
#Data Stat
dataset.iloc[:, 1].value_counts()

# =========================Cleaning Data ============================== 

# just keep english alphabet , spance and apostroph. remove all specail characters 
def remove_spectial_char(verse):
   
    cleaned_verse = ''.join(char for char in verse if char in alphapet)
    return cleaned_verse



#==================================Char counts for Padding===================== 
#Get maximum length to padding other verses with spaces 
data=dataset
data['char_count'] = [len(str(verse)) for verse in data['Verse']]

#get max count of chars 
max_count_of_chars=data['char_count'].max()

# ============================= Encoding Data ================================= 
def encode_verse_in_Binary(verse):
    """convert verse to matrix (t,n)
       n : one hot vector 28 dimantion
       t : max count of chars
    """
    #make verse as lower chares
    verse = verse.lower()
    #remove any strange char that don't in our alphabet
    verse = remove_spectial_char(verse)

    current_verse_length=len(verse)
    #Padding verse with spaces  
    verse=verse + ' '*(max_count_of_chars-current_verse_length)
    #encode each char with it's binary encoding in verse
    binary_encoded = [binary_alphapet_dict[char] for char in verse ]
    return  binary_encoded


def encode_meter(y_data):
    # integer encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform([meter for i,meter in enumerate(y_data)] )#shuffledData.iloc[:,1]

    # one hot encoding
    one_hot_encoded = to_categorical(integer_encoded)
    return one_hot_encoded
    

#Encode X_data : all verses
X = np.asarray([encode_verse_in_Binary(verse) for i,verse in enumerate(dataset.iloc[:,0])])
#the shape of encoding matrix
X.shape
#Encode Y_data 
Y = np.asarray(encode_meter(dataset.iloc[:,1]))
#the shape of encoding matrix
Y.shape
#=============================Save Encoding ===================================
def save(nameOfFile,nameOfDataset,dataVar):
    h5f = h5py.File(nameOfFile, 'w')
    h5f.create_dataset(nameOfDataset, data=dataVar)
    h5f.close()
    
#save("../data/new_data_matrix_X_binary_encoding.h5","X",X)
#save("../data/new_data_matrix_Y_one_hot_encoding.h5","Y",Y)

#---------------------------Retrive Encoding--------------------------------------
def restore (nameOfFile,nameOfDataset):
    h5f = h5py.File(nameOfFile,'r')
    matrix = h5f[nameOfDataset][:]
    h5f.close()
    return matrix
