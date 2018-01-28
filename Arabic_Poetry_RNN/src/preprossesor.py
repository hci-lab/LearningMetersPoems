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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Dropout,LSTM, Lambda,Bidirectional
from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed
import os,errno
import keras


def get_input_encoded_date(input_data_path,drop_cols,with_tashkeel_flag):
    input_arabic_poetry_dataset = pd.read_csv(input_data_path, sep = ",")
    input_arabic_poetry_dataset.drop(input_arabic_poetry_dataset.columns[drop_cols], axis=1,inplace=True)
    input_arabic_poetry_dataset.columns = ['Bayt_Text', 'Category']
    our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
    our_alphabets = "".join(our_alphabets)
    input_arabic_poetry_dataset['Bayt_Text'] = input_arabic_poetry_dataset['Bayt_Text'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(r'ّ+', 'ّ', x)).apply(lambda x: x.strip())
    
    if(with_tashkeel_flag == 0):        
        input_arabic_poetry_dataset['Bayt_Text'] = input_arabic_poetry_dataset['Bayt_Text'].apply(araby.strip_tashkeel).apply(araby.strip_tatweel)
    
    max_Bayt_length =  input_arabic_poetry_dataset.Bayt_Text.map(len).max()
    Bayt_Text_Encoded = input_arabic_poetry_dataset['Bayt_Text'].apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, max_Bayt_length))
    print("Input Data Bayt_Text encoded done.")
    Bayt_Text_Encoded_Stacked = np.stack(Bayt_Text_Encoded,axis = 0)    
    
    numbber_of_bohor = input_arabic_poetry_dataset['Category'].unique().size
    Bayt_Bahr_encoded = get_classes_encoded_date(input_arabic_poetry_dataset['Category'])
    return  Bayt_Text_Encoded_Stacked,Bayt_Bahr_encoded,max_Bayt_length,numbber_of_bohor


def get_classes_encoded_date(input_category_datasets):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(input_category_datasets)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Bayt_Bahr_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("Input Data Category encoded done.")
    return Bayt_Bahr_encoded

def get_model(num_layers_hidden,layers_type,n_units,max_Bayt_length,activation_output_function,numbber_of_bohor, load_weights_flag,checkpoints_path,last_or_max_val_acc):
    
    print("num_layers_hidden %d" %num_layers_hidden)
    print("layers_type %s" %layers_type)
    print("n_units %d" %n_units)
    print("max_Bayt_length %d" %max_Bayt_length)
    print("activation_output_function %s" %activation_output_function)
    print("numbber_of_bohor %d" %numbber_of_bohor)
    print("load_weights_flag %d" %load_weights_flag)
    print("checkpoints_path %s" %checkpoints_path)
    print("last_or_max_val_acc %d" %last_or_max_val_acc)
    
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
    #if(load_weights_flag == 1):
        #model = load_weights(load_weights_flag,checkpoints_path,last_or_max_val_acc,model)    
    
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
