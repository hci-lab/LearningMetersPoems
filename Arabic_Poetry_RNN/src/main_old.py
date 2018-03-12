#!/usr/bin/env python
# -*- coding: utf-8 -*-


###### TODO: List
# Read padded data from the files 
# Add option to read from padded or to rerun it again.
# =============================================================================

# =============================================================================
import os,errno
os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/MasterCode/ArabicPoetry-1/Arabic_Poetry_RNN/src/")
# =============================================================================


import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
from time import time
import keras
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Dropout,LSTM, Lambda,Bidirectional
from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed
#from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
from itertools import product 
import helpers
from helpers import string_with_tashkeel_vectorizer,string_vectorizer
import arabic
import pyarabic.araby as araby
import re

#from keras.layers.core import

print("Imports Done")
# =============================================================================
np.random.seed(7)

arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =======================Program Parameters====================================

load_weights_flag = 1
#Experiement_Name = 'Experiement_1_WITH_Tashkeel_ASIS'
Experiement_Name = 'Experiement_3_WITH_Tashkeel_ASIS_OldData_8bits_50units'
earlystopping_patience=-1  
test_size_param=0.1
validation_split_param = 0.1
n_units = 50
input_data_path = "../data/All_Data.csv"
# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 0
#input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"
epochs_param = 4
batch_size_param = 32
old_date_flag = 1
new_encoding_flag = 1
#===============================Concatinated Variables ========================

checkpoints_path ="../checkpoints/"+Experiement_Name+"/"
check_points_file_path = checkpoints_path+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
board_log_dir="../logs/"+Experiement_Name+"/"#+.format(time())

try:
    os.makedirs(board_log_dir)
    os.makedirs(checkpoints_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Can't create file for checkpoints or for logs please check ")
        raise
print("Input Parameters Defined and Experiement directory created")
# =========================Data Loading========================================

input_arabic_poetry_dataset = pd.read_csv(input_data_path, sep = ",")
if (old_date_flag == 1):
    print("working into old data sample ")
    cols = [1,2,4]
    input_arabic_poetry_dataset.drop(input_arabic_poetry_dataset.columns[cols], axis=1,inplace=True) 
   input_arabic_poetry_dataset.columns = ['Bayt_Text', 'Category']
    our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
    our_alphabets = "".join(our_alphabets)
    input_arabic_poetry_dataset['Bayt_Text'] = input_arabic_poetry_dataset['Bayt_Text'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(r'ّ+', 'ّ', x)).apply(lambda x: x.strip())
    
    input_arabic_poetry_dataset['Bayt_Text'] = input_arabic_poetry_dataset['Bayt_Text'].apply(araby.strip_tashkeel).apply(araby.strip_tatweel)
    #    
    
    if(new_encoding_flag ==0 ):
        Bayt_Text_Encoded = input_arabic_poetry_dataset['Bayt_Text'].apply(string_vectorizer)
    else:
        
        max_Bayt_length =  input_arabic_poetry_dataset.Bayt_Text.map(len).max()
        Bayt_Text_Encoded = input_arabic_poetry_dataset['Bayt_Text'].apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, max_Bayt_length))
    print("Input Data Bayt_Text encoded done.")

else:
    print("working on new data sample")
    cols = [0,1,2,3,4,6,7]
    input_arabic_poetry_dataset.drop(input_arabic_poetry_dataset.columns[cols], axis=1,inplace=True)
    input_arabic_poetry_dataset.columns = [ 'Category','Bayt_Text']
    Bayt_Text_Encoded = input_arabic_poetry_dataset['Bayt_Text'].apply(string_with_tashkeel_vectorizer)
    print("Input Data Bayt_Text encoded done.")

    
#input_arabic_poetry_dataset['Bayt_Text'] = input_arabic_poetry_dataset['Bayt_Text'].apply(araby.strip_tashkeel).apply(araby.strip_tatweel)
Bayt_Text_Encoded_Stacked = np.stack(Bayt_Text_Encoded,axis = 0)
max_Bayt_length =  input_arabic_poetry_dataset.Bayt_Text.map(len).max()

print("Input Data Read done.")


# =============================================================================
#one hot encoding for classes
# =============================================================================
Bayt_Bahr = input_arabic_poetry_dataset['Category']
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

print("Input Data Category encoded done.")
# =============================================================================


# =============================================================================
X_train, X_test, Y_train, Y_test=train_test_split(Bayt_Text_Encoded_Stacked, #bayts
                                                    Bayt_Bahr_encoded, #classes
                                                    test_size=test_size_param, 
                                                    random_state=0)
#default padding need to check the paramters details
print("Input Train/Test Split done.")
# =================================Padding=====================================

#X_train_padded = sequence.pad_sequences(X_train, maxlen=max_Bayt_length)
#X_test_padded = sequence.pad_sequences(X_test, maxlen=max_Bayt_length)

#print("Padding done.")
# =============================================================================


# =========================With Bi-LSTM Layers ================================
# create model
#K.set_learning_phase(1) #set learning phase
#
model = Sequential()


# Adding the input layer and the LSTM layer
if (new_encoding_flag == 0):
    model.add(LSTM(units = n_units, input_shape=(max_Bayt_length, numberOfUniqueChars), return_sequences=True))
else:
    model.add(LSTM(units = n_units, input_shape=(max_Bayt_length, 8), return_sequences=True))
    
#model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
#model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units, return_sequences=True))
#model.add(Dropout(0.1,seed=7)) 

model.add(LSTM(n_units))
#model.add(Dropout(0.1,seed=7)) 

#model.add(LSTM(n_units))
#model.add(Dropout(0.1,seed=7)) 
 

# Adding the output layer
model.add(Dense(units = numbber_of_bohor,activation = 'softmax'))


#===========================load weights====================================
if(load_weights_flag == 1):
    try:
        #List all avialble checkpoints into the directory
        checkpoints_path_list = os.listdir(checkpoints_path)
        all_checkpoints_list = [os.path.join(checkpoints_path,i) for i in checkpoints_path_list]
        #Get the last inserted weight into the checkpoint_path
        all_checkpoints_list_sorted = sorted(all_checkpoints_list, key=os.path.getmtime)
        if(last_or_max_val_acc == 0):
            print ("last check point")
            print(checkpoints_path+'weights-improvement-last-epoch.hdf5')
            max_weight_checkpoints = checkpoints_path+'weights-improvement-last-epoch.hdf5'#all_checkpoints_list_sorted[-1]
            #load weights
            model = keras.models.load_model(max_weight_checkpoints)
        else:
            print ("max_weight_checkpoints")
            all_checkpoints_list_sorted.remove(checkpoints_path+'weights-improvement-last-epoch.hdf5')
            epochs_list = [int(re.findall(r'-[0-9|.]*-',path)[0].replace('-',""))
                           for path in all_checkpoints_list_sorted]
            max_checkpoint = all_checkpoints_list_sorted[epochs_list.index(max(epochs_list))]
            print(max_checkpoint)
            #load weights
            model = keras.models.load_model(max_checkpoint)

        print (" max_weight_checkpoints")
        print(all_checkpoints_list_sorted[-1])
        max_weight_checkpoints =  all_checkpoints_list_sorted[-1]
        # load weights
        model.load_weights(max_weight_checkpoints)
    except IOError:
        print('An error occured trying to read the file.')
    except:
        if "last" not in  all_checkpoints_list_sorted[-1]:
            sys.exit("Last epoch don't exist in this modle , you can make last_or_max_val_acc=1 to load the epoch has max val_acc")
        else:
            print("No wieghts avialable \n check the paths")
        
# Compiling the RNN
model.compile(optimizer = 'adam', 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
print("Model Defined and compliled.")

#===========================last_epoch_saver====================================
class last_epoch_saver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #save last epoch weghits 
        self.model.save(checkpoints_path+"weights-improvement-last-epoch.hdf5")
        print("Save last epoch Done! ....")


checkpoint = ModelCheckpoint(check_points_file_path, 
                             monitor='val_acc', 
                             verbose=1,
                             save_best_only=True, 
                             mode='max')

print("Model checkpoint defined to track val_acc")
#===========================tensorboard========================================
tensorboard  = keras.callbacks.TensorBoard(log_dir=board_log_dir , 
                                           histogram_freq=0, 
                                           batch_size=batch_size_param, 
                                           write_graph=True, 
                                           write_grads=True, 
                                           write_images=True, 
                                           embeddings_freq=0, 
                                           embeddings_layer_names=None, 
                                           embeddings_metadata=None)

print("Model tensorboard defined")
#==============================================================================    
#
#===========================earlystopping======================================
earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                             min_delta=0,
                                             patience=earlystopping_patience,
                                             verbose=1,
                                             mode='auto')


last_epoch_saver_ = last_epoch_saver()

callbacks_list = [checkpoint,tensorboard]

if earlystopping_patience ==-1:
    callbacks_list = [checkpoint,tensorboard,last_epoch_saver_]
    print("Add  checkpoint - tensorboard - last_epoch_saver")
else:
    callbacks_list = [checkpoint,tensorboard,earlystopping,last_epoch_saver_]
    print("Add  checkpoint - tensorboard - earlystopping - last_epoch_saver")
#==============================================================================    
    
print(model.summary())

print("Model Training and validation started")

# Fitting the RNN to the Training set
#hist = model.fit(X_train_padded, 
#                 y_train, 
#                 validation_split = validation_split_param, 
#                 epochs=epochs_param, 
#                 batch_size=batch_size_param, 
#                 callbacks=callbacks_list,
#                 verbose=1)


#=============================Fitting Model====================================
# Fitting the RNN to the Training set
hist = model.fit(X_train, 
                 Y_train, 
                 validation_split = validation_split_param, 
                 epochs=epochs_param, 
                 batch_size=batch_size_param, 
                 callbacks=callbacks_list,
                 verbose=1)
#==============================================================================

#==============================================================================
#save last epoch weghits 
model.save(checkpoints_path+"weights-improvement-last-epoch.hdf5")
print("Save last epoch Done! ....")

#==============================================================================
print("Model Training and validation finished")
#print(history.losses)

#===========================Evaluate model=====================================
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

#==============================================================================


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