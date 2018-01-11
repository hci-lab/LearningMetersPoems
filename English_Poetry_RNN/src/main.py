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
from keras.layers import Dropout,LSTM, Lambda,Bidirectional,GRU
from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed
#from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys
 
# =============================================================================
np.random.seed(7)


# =========================Functions ==========================================
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
# =============================================================================


# =======================Program Parameters====================================
input_data_path = "./data/All_Data.csv"
load_weights_flag = 0
Experiement_Name = 'Experiement1'
layer_number = 1
#if u need one number for all layers add number alone
n_units = [500]
# 1->LSTM  , 2->GRU , 3->Bi-LSTM 
cell_mode = 1
drop_out_rate = 0.1
test_size_param=0.05
validation_split_param = 0.02
batch_size_param = 100
# 0 -> for test mode , 1 -> for train mode
learning_mode = 1

epochs_param = 20
#num of epoch should be wait when monitor don't change
earlystopping_patience=3  




#===============================Concatinated Variables ========================

checkpoints_path ="./Experiement/checkpoints/"+Experiement_Name+"/"
check_points_file_path = checkpoints_path+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
board_log_dir="./Experiement/logs/"+Experiement_Name+"/"#+.format(time())
losses_dir = "./Experiement/losses/"+Experiement_Name+"_ loss"# *************************************

try:
    os.makedirs(board_log_dir)
    os.makedirs(checkpoints_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Can't create file for checkpoints or for logs please check ")
        raise

    
# =========================Data Loading========================================
X,Y = load_encoding_data()#*********************************
max_Bayt_length=0#**************************
char_dimension=0#********************************
numbber_of_bohor=0#************************
# =============================================================================


# ==============================Split Data=====================================
X_train, X_test, Y_train, Y_test=train_test_split(X, #bayts
                                                  Y, #classes
                                                  test_size=test_size_param, 
                                                  random_state=0)
# =============================================================================


# =========================RNN models ================================
# create model
K.set_learning_phase(learning_mode) #set learning phase

model = Sequential()

# add layers
for n in range(layer_number):
    if len(n_units)==1 and layer_number>1:
        i=0
    else:
        if len(n_units)>=1 and len(n_units) != layer_number:
            sys.exit("pleas make length of n_units == layer_number or add only one element in n_units ")
        i=n
    
    # check if LSTM 
    if cell_mode==1:
        #check if first layer to add input_shape
        if n==0:
            # if NN as only one layer so shuld remove retun_sequences
            if layer_number == 1:
                model.add(LSTM(n_units[i],input_shape=(max_Bayt_length,char_dimension)))
            # if NN has many layers so should add return_sequences
            else:
                model.add(LSTM(n_units[i], return_sequences=True,input_shape=(max_Bayt_length,char_dimension)))
        # if it's not the first layer
        else:
            #check if last layer
            if len(n_units)-1 == i:
                model.add(LSTM(n_units[i]))
            else:
                model.add(LSTM(n_units[i], return_sequences=True))

    #check if GRU    
    elif cell_mode==2:

        #check if first layer to add input_shape
        if n==0:
            if layer_number  == 1:
            # if NN as only one layer so shuld remove retun_sequences
                model.add(GRU(n_units[i],input_shape=(max_Bayt_length,char_dimension)))
            # if NN has many layers so should add return_sequences
            else:
                model.add(GRU(n_units[i], return_sequences=True,input_shape=(max_Bayt_length,char_dimension)))
        # if it's not the first layer
        else:
            #check if last layer
            if len(n_units)-1 == i:
                model.add(GRU(n_units[i]))
            else:
                model.add(GRU(n_units[i], return_sequences=True))        
    #check if Bi-LSTM
    else:
        #check if first layer to add input_shape
        if n==0:
            # if NN as only one layer so shuld remove retun_sequences
            if layer_number  == 1:
                model.add(Bidirectional(LSTM(n_units[i]),
                                        input_shape=(max_Bayt_length, char_dimension)))
            # if NN has many layers so should add return_sequences
            else:
                model.add(Bidirectional(LSTM(n_units[i], return_sequences=True),
                                        input_shape=(max_Bayt_length, char_dimension)))
        # if it's not the first layer
        else:
            #check if last layer
            if len(n_units)-1 == i:
                model.add(Bidirectional(LSTM(n_units[i])))
            else:
                model.add(Bidirectional(LSTM(n_units[i], return_sequences=True)))
    #check if there Dopout or not
    if drop_out_rate != 0:
        model.add(Dropout(drop_out_rate))
                
        
#add softmax layer
model.add(Dense(units = numbber_of_bohor,activation = 'softmax'))



#==================================check to load last epoch====================
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


        

#==================================compile model===============================        
# Compiling the RNN
model.compile(optimizer = 'adam', 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])




#==============================Callbacks========================================
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

earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                             min_delta=0,
                                             patience=earlystopping_patience,
                                             verbose=1,
                                             mode='auto')

callbacks_list = [checkpoint,tensorboard,earlystopping]

print(model.summary())
#==============================================================================


#=============================Fitting Model====================================
# Fitting the RNN to the Training set
hist = model.fit(X_train, 
                 Y_train, 
                 validation_split = validation_split_param, 
                 epochs=epochs_param, 
                 batch_size=batch_size_param, 
                 callbacks=callbacks_list,
                 verbose=1)

#===========================Evaluate model=====================================
# Final evaluation of the model
scores = model.evaluate(X_test_padded, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))




