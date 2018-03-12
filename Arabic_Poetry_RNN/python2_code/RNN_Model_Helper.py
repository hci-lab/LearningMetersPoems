# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:52 2018

@author: Mostafa Alaa
"""
from __future__ import print_function

import os,sys
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional#
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from functools import partial, update_wrapper

#from keras.layers import Input, Lambda,Dropout
#from keras.preprocessing import sequence
#from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed

def wrapped_partial(func, *args, **kwargs):
    """ Function to handle this error AttributeError: 'functools.partial' 
    object has no attribute '__name__' """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


den_total_sample =  ((1/68601)  + ( 1/50574) + (1/37003 ) + ( 1/23238 ) + ( 1/17984 ) + ( 1/9075 ) + ( 1/7638 ) + ( 1/6662 ) + ( 1/4375 ) + ( 1/2085 ) + ( 1/380))

#=============================================================================
def w_categorical_crossentropy(y_true, y_pred,classes_dest,classes_encoder):
    """ # Custom loss function with costs """
    inverted = classes_encoder.inverse_transform([argmax(y_true)])
    #inverted = np.stack(inverted,axis = 0)
    n = classes_dest.loc[classes_dest['Bohor'] == inverted[0] , 'Cnt'].iloc[0]
#    return K.categorical_crossentropy(y_pred, y_true) * (1/n)
    return K.categorical_crossentropy(y_pred, y_true) * ((1/n) / den_total_sample ) 


#=============================================================================
def get_model(num_layers_hidden,layers_type,n_units,max_Bayt_length,activation_output_function, load_weights_flag,checkpoints_path,last_or_max_val_acc,label_encoder_output,classes_freq,weighted_loss_flag):
        
    numbber_of_bohor = classes_freq['Bohor'].unique().size
# =============================================================================
#     print("num_layers_hidden %d" %num_layers_hidden)
#     print("layers_type %s" %layers_type)
#     print("n_units %d" %n_units)
#     print("max_Bayt_length %d" %max_Bayt_length)
#     print("activation_output_function %s" %activation_output_function)
#     print("numbber_of_bohor %d" %numbber_of_bohor)
#     print("load_weights_flag %d" %load_weights_flag)
#     print("checkpoints_path %s" %checkpoints_path)
#     print("last_or_max_val_acc %d" %last_or_max_val_acc)
#     
# =============================================================================
    model = Sequential()
    print('Model Sequential defined')
    if(layers_type == 'LSTM'):
        print('Model LSTM Layer added')
        model.add(LSTM(units = n_units, input_shape=(max_Bayt_length, 8), return_sequences=True))
    elif  (layers_type == 'Bidirectional_LSTM'):
        print('Model input Bidirectional_LSTM Layer added')
        model.add(Bidirectional(LSTM(n_units, return_sequences=True), input_shape=(max_Bayt_length, 8)))
    else:
        print('Model Dense Layer added')
        model.add(Dense(n_units,activation = 'relu',input_shape=(max_Bayt_length, 8)))
    
    for _ in range(num_layers_hidden-1):
        if(layers_type == 'LSTM'):
            print('Model LSTM Layer added')
            model.add(LSTM(n_units, return_sequences=True))
        elif  (layers_type == 'Bidirectional_LSTM'):
            print('Model Bidirectional_LSTM Layer added')
            model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
        else:
            print('Model Dense Layer added')
            model.add(Dense(n_units))
    
    if(layers_type == 'LSTM'):
        print('Model LSTM prefinal Layer added')
        model.add(LSTM(n_units))
    elif  (layers_type == 'Bidirectional_LSTM'):
        print('Model Bidirectional_LSTM prefinal Layer added')
        model.add(Bidirectional(LSTM(n_units)))
    else:
        print('Model Dense prefinal Layer added')
        model.add(Dense(n_units))
        
    # Adding the output layer
    print('Model Dense final Layer added')
    model.add(Dense(units = numbber_of_bohor,activation = activation_output_function))
    if(load_weights_flag == 1):
        print("Loading Old model")
        model = load_weights(load_weights_flag,checkpoints_path,last_or_max_val_acc,model)    
    w_categorical_crossentropy_Pfun = wrapped_partial(w_categorical_crossentropy, classes_dest = classes_freq,classes_encoder = label_encoder_output)

    
    if(weighted_loss_flag == 1):
        model.compile(optimizer = 'adam', loss=w_categorical_crossentropy_Pfun, metrics = ['accuracy'])
    else:
        model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    print('Model Compiled')
    print(model.summary())
    
    return model
    
def load_weights(load_weights_flag,checkpoints_path,last_or_max_val_acc,model):
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

    return model