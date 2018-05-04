#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================

#os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/ArabicPoetry
# /ArabicPoetry-1/Arabic_Poetry_RNN/src/")
#os.chdir("/media/mostafaalaa/Main_Hard/Learning/Master/CombinedWorkspace/Python
# /DeepLearningMaster/ArabicPoetry/ArabicPoetry-1/Arabic_Poetry_RNN/src")
# =============================================================================


# Multi_GPU_Flag
MULTI_GPU_FLAG = False

import os
import sys

# before Keras / Tensorflow is imported.
if len(sys.argv) == 2 and sys.argv[1] == '--cpu':
    print('Running On CPU')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    MULTI_GPU_FLAG = False

# before Keras / Tensorflow is imported.
if len(sys.argv) == 2 and sys.argv[1] == '--multgpu':
    MULTI_GPU_FLAG = True

from sys import path
# Relative path to this modul's location in PyQuran.
searchingPath = 'lib'

# The current path of the current module.
current_path  = os.path.dirname(os.path.abspath(__file__))
# Joining this module's path with the relative path of the corpus
path_ = os.path.join(current_path, searchingPath)
path.append(path_)

import numpy as np
import arabic
import random as rn
from tensorflow import set_random_seed
from Exp_Runner import runner
import warnings
import comparing_
import helpers
import tensorflow as tf

# ==============================================================================


print("Imports Done")

# ===================Tensorflow Config and Gpu settings=========================
import tensorflow as tf
tf.reset_default_graph()
with tf.Graph().as_default():
    set_random_seed(123456)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
print("Tensorflow Config and Gpu settings Done")

# =====================Random seeds settings ===================================
np.random.seed(123456)
rn.seed(123456)
print("Random seeds settings Done")
# =============================================================================

# =============================================================================
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)
# =============================================================================


# =======================Program Parameters====================================
# =============================================================================
layers_type = ["Bidirectional_LSTM" , "LSTM"]
num_layers_hidden = ["3","6"]
weighted_loss_flag = ["0","1"]
n_units = ["50","82"]
encoded_X_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5",
                   "../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5",
                   "../data/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5",
                   "../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]


encoded_Y_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5",
                   "../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5",
                   "../data/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5",
                   "../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]

full_classes_encoder_path = "../data/Encoded/8bits/encoders_full_dat.pickle"
eliminated_classes_encoder_path = "../data/Encoded/8bits/encoders_eliminated_data.pickle"

epochs_param = 40
# umar -> it wasn't found
batch_size_param = 2048

activation_output_function = 'softmax'
# umar -> not used
#new_encoding_flag = 1
earlystopping_patience = -1
# umar -> not used
#required_data_col = [0, 2, 3, 5]
test_size_param = 0.1
validation_split_param = 0.1

# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 1
load_weights_flag = 0

# =============================================================================

# =============================================================================
## in case testing un comment the below and comment the above block
#layers_type = ["Bidirectional_LSTM"]
#num_layers_hidden = ["3"]
#weighted_loss_flag = ["1"]
#n_units = ["50"]
#batch_size_param = 512
#encoded_X_paths = ["../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]
#encoded_Y_paths = ["../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]
#epochs_param = 20

# =============================================================================


#input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"




# Remove the mess
def removeTestingFiles():
    rmCommand = 'rm -rf'
    rmCommand += ' lib/test_folders/*'
    rmCommand += ' log.txt'
#    rmCommand += ' All_Experiments_Results.txt'
    # Executing the command:
    if os.system(rmCommand) == 0:
        print('old testing files are removed!')
    else:
        print('there is no testing files to remove!')

# Command line arguemnt option; --test
'''
    TODO
        0 Removes old testing paths (not encoding h5)
        2 Remove log.txt, Results, Checkpoints, Logs
        3 Return True | False
        4 Remove the testing Results, Checkpoints, Logs
'''
test = False
if len(sys.argv) == 2 and sys.argv[1] == '--test':
    test = True
    # Removing the mess (if there were!)
    removeTestingFiles()

    # Testing Setup
    layers_type = ["LSTM"]
    num_layers_hidden = ["1"]
    weighted_loss_flag = ["1"]
    n_units = ["1"]
    encoded_X_paths = [
    "lib/data_testing/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5",
    "lib/data_testing/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5",
    "lib/data_testing/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5",
    "lib/data_testing/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]
    encoded_Y_paths = [
    "lib/data_testing/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5",
    "lib/data_testing/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5",
    "lib/data_testing/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5",
    "lib/data_testing/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]
    epochs_param = 1
    batch_size_param = 1024
    last_or_max_val_acc = 1
    activation_output_function = 'softmax'
    earlystopping_patience = -1
    test_size_param = 0.1
    validation_split_param = 0.1
    load_weights_flag = 0
    full_classes_encoder_path = "lib/data_testing/encoders_full_dat.pickle"
    eliminated_classes_encoder_path = "lib/data_testing/encoders_eliminated_data.pickle"




def checking_experiment_run(exp_name):
    '''
    this function is responsible for continue experiments running if running
    stop for any reson. 
    it will make file start from experiment that cut off.
    '''
    try:
        file = open('log.txt','r+')
    except:
        file = open('log.txt','w+')
    lines = {}
    line = file.readline()
    while line:
        parts = line.split(",")
        exp_n , exp_state = parts[0] , parts[1]
        lines[exp_n] = exp_state
        line = file.readline()
    if exp_name not in lines.keys():
        file.write(exp_name+","+"running@0\n")
        file.close()
        return "added",0
    else:
        state, last_epoch = lines[exp_name].split('@')[0] , lines[exp_name].split('@')[1] 
        return state , int(last_epoch.split('\n')[0])

    
#save name of previous experiment
previous_experiment_name = ""

# ===============================Concatinated Variables ========================
counter = 0
for X, Y in zip(encoded_X_paths, encoded_Y_paths):
    file_name = X.split("/")[7 -1].split(".")[0]
    # print(file_name)
    for l_type in layers_type:
        exp_l_type = file_name + "_" + l_type
        # print(exp_l_type)
        for l_num in num_layers_hidden:
            exp_l_num = exp_l_type + "_" + l_num
            for units in n_units:
                exp_unit = exp_l_num + "_" + units
                for w_flag in weighted_loss_flag:
                    new_load_weights_flag = load_weights_flag
                    new_last_or_max_val_acc = last_or_max_val_acc
                    new_epochs_param = epochs_param
                    exp_w_flag = exp_unit + "_" + w_flag
                    counter += 1
                    Experiement_Name = "Exp_" + str(counter) + "_" + exp_w_flag
                    print(Experiement_Name)
                    exp_state,last_epoch = checking_experiment_run(Experiement_Name)
                    #check if this not exp finished
                    if exp_state != 'done':
                        #check if exp exisit but not finished
                        if exp_state == 'running':
                           print("contnue expriment :: ",Experiement_Name,"  from epoch : ",last_epoch)
                           # to load modle  
                           new_load_weights_flag = 1
                           # to load last epoch wights
                           new_last_or_max_val_acc = 0
                           #claculate remain epoch num
                           new_epochs_param = new_epochs_param - last_epoch
                           if last_epoch == 0:
                               new_load_weights_flag = 0
                        runner(X,
                               Y,
                               test_size_param,
                               int(l_num),
                               l_type,
                               validation_split_param,
                               new_epochs_param,
                               int(units),
                               activation_output_function,
                               new_load_weights_flag,
                               new_last_or_max_val_acc,
                               int(w_flag),
                               batch_size_param,
                               earlystopping_patience,
                               Experiement_Name,
                               full_classes_encoder_path,
                               eliminated_classes_encoder_path,
                               path_,
                               MULTI_GPU_FLAG,
                               test)

                        #update experiment_state to done 
                        helpers.update_log_file(Experiement_Name,"done@0", False)
                        
                    print(Experiement_Name,"   Done !!")
                    print("=" * 80)

if len(sys.argv) == 2 and sys.argv[1] == '--test':

    print('End Testing '*20)
    os.system('clear')
    print(comparing_.check_results())
    removeTestingFiles()
