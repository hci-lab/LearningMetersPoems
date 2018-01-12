#!/usr/bin/env python
# -*- coding: utf-8 -*-


###### TODO: List
# Read padded data from the files 
# Add option to read from padded or to rerun it again.
# =============================================================================
import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
import os,errno
from time import time
import keras
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Dropout,LSTM, Lambda,Bidirectional
from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed
#from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras import backend as K
from itertools import product 
import helpers
#from keras.layers.core import

# =============================================================================
np.random.seed(7)
os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/GP-Ripo-master/Arabic_Poetry_RNN/")

import src.arabic as arabic
#import src.pyarabic.araby as araby
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =========================Functions ==========================================

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
# =============================================================================

def string_vectorizer(strng, alphabet=arabic_alphabet):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return array(vector)


def string_with_tashkeel_vectorizer(string, tashkeel=arabic.shortharakat):
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
    string = helpers.factor_shadda_tanwin(string)

    string_clean = [] # harakah is concatinated with previous latter.
    i = 0
    while True:
        # it is the last item?!
        if i == len(string) - 1:
            string_clean.append(string[i])
            break
        elif i > len(string)-1:
            break 
        elif string[i+1] not in tashkeel:        
            string_clean.append(string[i])
            i += 1
        elif string[i+1] in tashkeel:        
            string_clean.append(string[i] + string[i+1])
            i += 2


    # 1* Building letter and taskell compinations
    arabic_alphabet_tashkeel = helpers.lettersTashkeelCombination

    encoding_combination = array(helpers.encoding_combination)

    # 4* getting encoding for each letter from input string
    representation = []
    for x in string_clean:
        index = string_clean.index(x)
        representation.append(encoding_combination[index])

    return array(representation)
# =======================Program Parameters====================================

load_weights_flag = 0
Experiement_Name = 'Experiement_1_WITH_Tashkeel_ASIS'
test_size_param=0.1
validation_split_param = 0.05
n_units = 200
#input_data_path = "./data/All_Data.csv"
input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"
epochs_param = 20
batch_size_param = 64
#===============================Concatinated Variables ========================

checkpoints_path ="./checkpoints/"+Experiement_Name+"/"
check_points_file_path = checkpoints_path+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
board_log_dir="./logs/"+Experiement_Name+"/"#+.format(time())

try:
    os.makedirs(board_log_dir)
    os.makedirs(checkpoints_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Can't create file for checkpoints or for logs please check ")
        raise

# =========================Data Loading========================================
sample_arabic_poetry = pd.read_csv(input_data_path, sep = ",")
cols = [0,1,2,3,4,6,7]
sample_arabic_poetry.drop(sample_arabic_poetry.columns[cols], axis=1,inplace=True)
sample_arabic_poetry.columns = [ 'Category','Bayt_Text']
#sample_arabic_poetry['Bayt_Text'] = sample_arabic_poetry['Bayt_Text'].apply(araby.strip_tashkeel).apply(araby.strip_tatweel)
max_Bayt_length =  sample_arabic_poetry.Bayt_Text.map(len).max()
# =============================================================================




# =============================================================================
#Bayt_Text_Encoded = sample_arabic_poetry['Bayt_Text'].apply(string_vectorizer)
Bayt_Text_Encoded = sample_arabic_poetry['Bayt_Text'].apply(string_with_tashkeel_vectorizer)

# =============================================================================
#one hot encoding for classes
# =============================================================================
Bayt_Bahr = sample_arabic_poetry['Category']
numbber_of_bohor = Bayt_Bahr.unique().size

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
X_train, X_test, y_train, y_test=train_test_split(Bayt_Text_Encoded, #bayts
                                                    Bayt_Bahr_encoded, #classes
                                                    test_size=test_size_param, 
                                                    random_state=0)
#default padding need to check the paramters details

X_train_padded = sequence.pad_sequences(X_train, maxlen=max_Bayt_length)
X_test_padded = sequence.pad_sequences(X_test, maxlen=max_Bayt_length)
# =============================================================================


# =========================With Bi-LSTM Layers ================================
# create model
K.set_learning_phase(1) #set learning phase

n_units = 100
model = Sequential()


# Adding the input layer and the LSTM layer

model.add(LSTM(units = n_units, input_shape=(max_Bayt_length, numberOfUniqueChars), return_sequences=True))
model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
model.add(Dropout(0.1,seed=7)) 
 

# Adding the output layer
model.add(Dense(units = numbber_of_bohor,activation = 'softmax'))

# load weights
if(load_weights_flag == 1):
    try:
        #List all avialble checkpoints into the directory
        checkpoints_path_list = os.listdir(checkpoints_path)
        all_checkpoints_list = [os.path.join(checkpoints_path,i) for i in checkpoints_path_list]
        #Get the last inserted weight into the checkpoint_path
        all_checkpoints_list_sorted = sorted(all_checkpoints_list, key=os.path.getmtime)
        print (" max_weight_checkpoints")
        print(all_checkpoints_list_sorted[-1])
        max_weight_checkpoints =  all_checkpoints_list_sorted[-1]
        # load weights
        model.load_weights(max_weight_checkpoints)
    except IOError:
        print('An error occured trying to read the file.')
    except:
        print("No wieghts avialable \n check the paths")

        
# Compiling the RNN
model.compile(optimizer = 'adam', 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

print(model.summary())

checkpoint = ModelCheckpoint(check_points_file_path, 
                             monitor='val_acc', 
                             verbose=1,
                             save_best_only=True, 
                             mode='max')

tensorboard  = keras.callbacks.TensorBoard(log_dir=board_log_dir , 
                                           histogram_freq=0, 
                                           batch_size=50, 
                                           write_graph=True, 
                                           write_grads=True, 
                                           write_images=True, 
                                           embeddings_freq=0, 
                                           embeddings_layer_names=None, 
                                           embeddings_metadata=None)

callbacks_list = [checkpoint,tensorboard]


#history = LossHistory()

# Fitting the RNN to the Training set
hist = model.fit(X_train_padded, 
                 y_train, 
                 validation_split = validation_split_param, 
                 epochs=epochs_param, 
                 batch_size=batch_size_param, 
                 callbacks=callbacks_list,
                 verbose=1)

print(history.losses)

# Final evaluation of the model
scores = model.evaluate(X_test_padded, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# ===========================Ploting===========================================
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
# =============================================================================
# model.add(Bidirectional(LSTM(n_units, return_sequences=True), input_shape=(max_Bayt_length, numberOfUniqueChars)))
# model.add(Dropout(0.1))
# 
# model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
# #model.add(Bidirectional(LSTM(n_units)))
# model.add(Dropout(0.1))
# 
# #model.add(Bidirectional(LSTM(n_units)))
# model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
# model.add(Dropout(0.1))
# 
# 
# model.add(Bidirectional(LSTM(n_units)))
# model.add(Dropout(0.1))
# 
# #model.add(Bidirectional(LSTM(n_units)))
# 
# #model.add(Bidirectional(LSTM(n_units)))
# #model.add(TimeDistributedDense(output_dim=5))
# 
# #model.add(TimeDistributed(Dense(11, activation='relu')))
# 
# 
# =============================================================================
# =============================================================================
# =========================With three hidden Layer=============================
# # create model
# n_units = 400
# 
# 
# model = Sequential()
# 
# Adding the input layer and the LSTM layer
#model.add(LSTM(units = n_units, input_shape = ( 82, 35), return_sequences=True))
# 
# model.add(LSTM(n_units, return_sequences=True))
# 
# model.add(LSTM(n_units, return_sequences=True))
# 
# model.add(LSTM(n_units, return_sequences=True))
# 
# model.add(LSTM(n_units, return_sequences=True))
# 
# model.add(LSTM(n_units, return_sequences=True))
# 
# model.add(LSTM(n_units))
# 
# 
# # Adding the output layer
# model.add(Dense(units = 11,activation = 'softmax'))
# 
# 
# # load weights
# #load the weights if the parameters == 1 else ignore load weights
# #check the exp folder path 
# #get the max path
# #model.load_weights("./checkpoints/Experiement6/weights-improvement-18-0.96.hdf5")
# 
# 
# # Compiling the RNN
# model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics = ['accuracy'])
# 
# print(model.summary())
# 
# #new_model = load_model("./checkpoints/Experiement6/weights-improvement-31-0.96.hdf5")
# 
# 
# filepath="./checkpoints/Experiement8/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
# 
# 
# # ===============================tensorboard===================================
# 
# =============================================================================

# =============================================================================

#from keras.models import load_model
#model = load_model('my_model.h5')
#lstmweights=model.get_weights()
#model2.set_weights(lstmweights)


#https://machinelearningmastery.com/check-point-deep-learning-models-keras/
# =============================================================================
# if (read_from_checkpoints == 1 and  checkpoint_best_only == 1 ):
#     print('Model Will read from Check points with Best Result only')
#     model.load_weights("weights.best.hdf5")
# 
# if (read_from_checkpoints == 1 and  checkpoint_best_only == 0 ):
#     print('Model Will read from Check points from the checkpoints files')
#     model.load_weights("weights.best.hdf5")
# 
# if(checkpoint_best_only == 1 and read_from_checkpoints == 0):
# # checkpoint
#     print('Model Will Train and save the best results only')
#     filepath="weights.best.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint]
# 
# if(checkpoint_best_only == 0 and read_from_checkpoints == 0):
#     print('Model Will Train and save the all wights results only')
#     filepath="weights-improvement-{epoch:20}-{three_layer}-{units:500}.hdf5"
#     #filepath="weights.best.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint]
# 
# =============================================================================


# =========================With four hidden Layer================================

# =============================================================================
# model = Sequential()
# 
# model.add(LSTM(units = 500, input_shape = ( 82, 35)))
# # Adding the one hidden layer
# model.add(Dense(units = 500,activation = 'relu'))
# 
# model.add(Dense(units = 500,activation = 'relu'))
# 
# model.add(Dense(units = 500,activation = 'relu'))
# 
# model.add(Dense(units = 500,activation = 'relu'))
# 
# # Adding the output layer
# model.add(Dense(units = 11,activation = 'softmax'))
# 
# # Compiling the RNN
# model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics = ['accuracy'] )
# 
# print(model.summary())
# 
# model.fit(X_train_padded, y_train, validation_split = 0.02, epochs=100, batch_size=20, callbacks=callbacks_list)
# 
# # Final evaluation of the model
# scores = model.evaluate(X_test_padded, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# 
# =============================================================================

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
# =============================================================================
#run on CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 
# =============================================================================
