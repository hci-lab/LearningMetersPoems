# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:52 2018

@author: Mostafa Alaa
"""
from __future__ import print_function

import os
import sys
import keras
import numpy as np
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import backend as K
# from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from functools import partial, update_wrapper


# =============================================================================
def wrapped_partial(func, *args, **kwargs):
    """ Function to handle this error AttributeError: 'functools.partial' 
    object has no attribute '__name__' """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


# =============================================================================


def w_categorical_crossentropy(y_true, y_pred, classes_dest, encoder, sum_of_classes_denesity):
    """ # Custom loss function with costs """

    inverted = encoder.inverse_transform([argmax(y_true)])
    inverted = np.stack(inverted, axis=0)
    sample_per_class = classes_dest.loc[classes_dest['Class'] == inverted[0], 'Cnt'].iloc[0]
    return K.categorical_crossentropy(y_true, y_pred) * (
            (1 / sample_per_class) / sum_of_classes_denesity)


# =============================================================================

# type(classes_dest.loc[classes_dest['Class'] == "الطويل" , 'Cnt'].iloc[0])

# =============================================================================


def get_model(num_layers_hidden,
              layers_type,
              n_units,
              max_bayt_length,
              encoding_length,
              activation_output_function,
              load_weights_flag,
              checkpoints_path,
              last_or_max_val_acc,
              weighted_loss_flag,
              classes_dest,
              classes_encoder):
    numbber_of_bohor = classes_dest.shape[0]  # classes_freq['Bohor'].unique().size

    # =============================================================================
    #     max_bayt_length = Bayt_Text_Encoded_Stacked.shape[1]
    #     encoding_length = Bayt_Text_Encoded_Stacked.shape[2]
    #      num_layers_hidden = int( num_layers_hidden[0])
    #      n_units= int( n_units[0])
    #      weighted_loss_flag= int( weighted_loss_flag[0])
    #      layers_type=layers_type[0]
    # =============================================================================

    print("num_layers_hidden %d" % num_layers_hidden)
    print("layers_type %s" % layers_type)
    print("n_units %d" % n_units)
    print("max_bayt_length %d" % max_bayt_length)
    print("max_bayt_length %d" % encoding_length)
    print("activation_output_function %s" % activation_output_function)
    print("load_weights_flag %d" % load_weights_flag)
    print("checkpoints_path %s" % checkpoints_path)
    print("last_or_max_val_acc %d" % last_or_max_val_acc)
    print("weighted_loss_flag %d" % weighted_loss_flag)
    print("numbber_of_bohor %d" % numbber_of_bohor)
    # =============================================================================

    model = Sequential()
    print('Model Sequential defined')
    if layers_type == 'LSTM':
        print('Model LSTM Layer added')
        model.add(LSTM(units=n_units, input_shape=(max_bayt_length, encoding_length),
                       return_sequences=True))
    elif layers_type == 'Bidirectional_LSTM':
        print('Model input Bidirectional_LSTM Layer added')
        model.add(Bidirectional(LSTM(n_units, return_sequences=True),
                                input_shape=(max_bayt_length, encoding_length)))
    else:
        print('Model Dense Layer added')
        model.add(Dense(n_units, activation='relu', input_shape=(max_bayt_length, encoding_length)))
    # =============================================================================
    for _ in range(num_layers_hidden - 1):
        if layers_type == 'LSTM':
            print('Model LSTM Layer added')
            model.add(LSTM(n_units, return_sequences=True))
        elif layers_type == 'Bidirectional_LSTM':
            print('Model Bidirectional_LSTM Layer added')
            model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
        else:
            print('Model Dense Layer added')
            model.add(Dense(n_units))
    # =============================================================================
    if layers_type == 'LSTM':
        print('Model LSTM prefinal Layer added')
        model.add(LSTM(n_units))
    elif layers_type == 'Bidirectional_LSTM':
        print('Model Bidirectional_LSTM prefinal Layer added')
        model.add(Bidirectional(LSTM(n_units)))
    else:
        print('Model Dense prefinal Layer added')
        model.add(Dense(n_units))
    # =============================================================================
    # Adding the output layer
    print('Model Dense final Layer added')
    model.add(Dense(units=numbber_of_bohor, activation=activation_output_function))
    # =============================================================================
    print("define partial function w_categorical_crossentropy_pfun")
    sum_of_classes_denesity = classes_dest.Cnt.apply(lambda x: 1 / x).sum()
    w_categorical_crossentropy_pfun = wrapped_partial(w_categorical_crossentropy,
                                                      classes_dest=classes_dest,
                                                      encoder=classes_encoder,
                                                      sum_of_classes_denesity=sum_of_classes_denesity)

    print("partial function w_categorical_crossentropy_pfun defined")
    # =============================================================================

    if load_weights_flag == 1:
        print("Loading Old model")
        model = load_weights(checkpoints_path, last_or_max_val_acc, weighted_loss_flag,
                             w_categorical_crossentropy_pfun)
    # =============================================================================
    if weighted_loss_flag == 1:
        print("Model w_categorical_crossentropy_pfun loss function defined")
        model.compile(optimizer='adam',
                      loss=w_categorical_crossentropy_pfun,
                      metrics=['accuracy'])

        print("Model w_categorical_crossentropy_pfun loss function finish")
    else:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    print('Model Compiled')
    print(model.summary())
    return model


# =============================================================================

# =============================================================================


# =============================================================================
# =============================================================================


def load_weights(checkpoints_path, last_or_max_val_acc, weighted_loss_flag,
                 w_categorical_crossentropy_pfun):
    try:
        # List all avialble checkpoints into the directory
        checkpoints_path_list = os.listdir(checkpoints_path)
        all_checkpoints_list = [os.path.join(checkpoints_path, i) for i in checkpoints_path_list]
        # Get the last inserted weight into the checkpoint_path
        all_checkpoints_list_sorted = sorted(all_checkpoints_list, key=os.path.getmtime)
        if last_or_max_val_acc == 0:
            print("last check point")
            print(checkpoints_path + 'weights-improvement-last-epoch.hdf5')
            max_weight_checkpoints = checkpoints_path + 'weights-improvement-last-epoch.hdf5'
            if weighted_loss_flag == 1:
                model_loaded = keras.models.load_model(max_weight_checkpoints, custom_objects={
                    "w_categorical_crossentropy": w_categorical_crossentropy_pfun})
            else:
                model_loaded = keras.models.load_model(max_weight_checkpoints)

        else:
            print("max_weight_checkpoints")
            all_checkpoints_list_sorted.remove(
                checkpoints_path + 'weights-improvement-last-epoch.hdf5')
            epochs_list = [int(re.findall(r'-[0-9|.]*-', path)[0].replace('-', ""))
                           for path in all_checkpoints_list_sorted]
            max_checkpoint = all_checkpoints_list_sorted[epochs_list.index(max(epochs_list))]
            print(max_checkpoint)
            # load weights
            if weighted_loss_flag == 1:
                model_loaded = keras.models.load_model(max_checkpoint, custom_objects={
                    "w_categorical_crossentropy": w_categorical_crossentropy_pfun})
            else:
                model_loaded = keras.models.load_model(max_checkpoint)

    # =============================================================================
    #         print (" max_weight_checkpoints")
    #         print(all_checkpoints_list_sorted[-1])
    #         max_weight_checkpoints =  all_checkpoints_list_sorted[-1]
    #         # load weights
    #         model.load_weights(max_weight_checkpoints)
    # =============================================================================
    except IOError:
        print('An error occured trying to read the file.')
    except:
        if "last" not in all_checkpoints_list_sorted[-1]:
            sys.exit(
                "Last epoch don't exist in this modle , you can make last_or_max_val_acc=1 to "
                "load the epoch has max val_acc")
        else:
            print("No wieghts avialable \n check the paths")

    return model_loaded


# =============================================================================
# =============================================================================


def recall_precision_f1(confusion_matrix_df):
    """Evaluating the model with Recall, Precision and F1 Score"""
    '''
    Args:
        confusion_matrix_df: a datafram with index_col=0
        
    returns: (x, y)
        x: is a datafrom of recall and precision for every class
        y: is the f1 score for the model.
        
    '''

    confusion_matrix_np = confusion_matrix_df.values
    bahr = 0
    sum_rows = []
    sum_columns = []
    diagonal_recall = []
    diagonal_precision = []

    matrices = [confusion_matrix_np, np.transpose(confusion_matrix_np)]
    flag = 0
    for matrix in matrices:
        # print(matrix)
        for x in matrix:
            # class_num is the sum of the ith row of the confusion matrix
            # Also it it the sum of the ith column, when flag = 1
            class_sum = np.sum(x)
            # Recall
            if flag == 0:
                sum_rows.append(class_sum)
                diagonal_recall.append(x[bahr])
            # Precision
            elif flag == 1:
                sum_columns.append(class_sum)
                diagonal_precision.append(x[bahr])

            bahr += 1
        flag += 1
        bahr = 0
    '''
    # Recall per class
    print(np.array(diagonal_recall)/ np.array(sum_rows))
    # Precision per class
    print(np.array(diagonal_precision)/ np.array(sum_columns))
    '''
    recall_per_class = np.array(diagonal_recall) / np.array(sum_rows)
    precision_per_class = np.array(diagonal_precision) / np.array(sum_columns)

    recall_mean = np.mean(recall_per_class)
    precision_mean = np.mean(precision_per_class)

    sum_rec_pre = recall_mean + precision_mean
    mul_rec_re = recall_mean * precision_mean
    f1_score = 2 * (mul_rec_re / sum_rec_pre)

    # Building the Data Frame
    result_dict = dict(Recall=recall_per_class, Precision=precision_per_class)
    results_DF = pd.DataFrame(result_dict, dtype=float)
    results_DF.index = confusion_matrix_df.index

    return results_DF, f1_score
