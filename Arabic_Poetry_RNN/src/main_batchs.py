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
use_CPU = False
fraction = 0.3
import os
import sys

if "--cpu" in sys.argv and  "--multgpu" in sys.argv:
    print("can't work with this two options together pleas set on of them :)")
    sys.exit()

# before Keras / Tensorflow is imported.
if '--cpu' in sys.argv:
    print('Running On CPU')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    MULTI_GPU_FLAG = False
    use_CPU = True
# before Keras / Tensorflow is imported.
elif '--multgpu' in sys.argv:
    MULTI_GPU_FLAG = True

if '--frac' in sys.argv:
    error = False
    try:
        fraction = float(sys.argv[sys.argv.index('--frac') + 1])
    except:
        error = True
    if error:
        print("pleas enter float number bettwen 0 < frac < 1 after --frac options")
        sys.exit()
    if fraction > 1 or fraction < 0.1:
        print("pleas enter float number bettwen 0 < frac < 1")
        sys.exit()

    
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
from keras.backend.tensorflow_backend import set_session
from Exp_Runner_batches import runner
import warnings
import comparing_
import helpers
import tensorflow as tf
import pandas as pd
import math
sys.path.append("lib/")
import helpers
import pyarabic
# ==============================================================================

print("Imports Done")

# ===================Tensorflow Config and Gpu settings=========================
import tensorflow as tf
tf.reset_default_graph()
with tf.Graph().as_default():
    set_random_seed(123456)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = fraction
set_session(tf.Session(config=config))
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
layers_type = ["Bidirectional_LSTM", "LSTM"]
num_layers_hidden = ["3","6"]
weighted_loss_flag = ["0","1"]
n_units = ["50","82"]

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

vectoriz_fun = ''
encoding_mark = ''
if '--encoding' in sys.argv:
    error = False
    vectorize_funs = [helpers.string_with_tashkeel_vectorizer_OneHot,
                      helpers.string_with_tashkeel_vectorizer,
                      helpers.two_hot_encoding]
    options = ['onehot', '8bit', 'twohot']
    try:
        option = sys.argv[sys.argv.index('--encoding') + 1]
        if option not in options:
            error = True
    except:
            option = 'onehot'
    if error:
        print('pleas enter one option after --encoding :- ')
        print("exmple :-  python3 filename <option> <sub-option of main option>")
        print("onehot :  for encode as onehot")
        print("twohot :  for encode as towhot")
        print("8bit   :  for encode as 8bit")
        sys.exit()

    if option == options[0]:
        vectoriz_fun = vectorize_funs[0]
        encoding_mark = "_one_hot_encoding"
    if option == options[1]:
        vectoriz_fun = vectorize_funs[1]
        encoding_mark = '_8bitsEncoding'
    if option == options[2]:
        vectoriz_fun = vectorize_funs[2]
        encoding_mark = '_two_hot_encoding'
else:
    vectoriz_fun = helpers.string_with_tashkeel_vectorizer_OneHot
    encoding_mark = "_one_hot_encoding"


expermen_names = []
if "--run" in sys.argv:
    error = False
    options = ['all', '1', '2', '3', '4']
    try:
        option = sys.argv[sys.argv.index('--run') + 1]
        if option not in options:
            error = True
    except:
            option = 'all'

    if error:
        print('pleas enter one option after --run :- ')
        print("exmple  python file <option> <sub-option of main option>")
        print("all :  for run all data types")
        print(" 1  :  for run only eliminated_data_matrix_without_tashkeel")
        print(" 2  :  for run only full_data_matrix_without_tashkeel")
        print(" 3  :  for run only eliminated_data_matrix_with_tashkeel")
        print(" 4  :  for run only full_data_matrix_with_tashkeel")
        sys.exit()
        
    if option == options[0]:
        expermen_names = ["eliminated_data_matrix_without_tashkeel"+ encoding_mark,
                          "full_data_matrix_without_tashkeel"+encoding_mark,
                          "eliminated_data_matrix_with_tashkeel"+encoding_mark,
                          "full_data_matrix_with_tashkeel"+encoding_mark]
    elif option == options[1]:
        expermen_names = ["eliminated_data_matrix_without_tashkeel"+encoding_mark]        
    elif option == options[2]:
        expermen_names = ["full_data_matrix_without_tashkeel"+encoding_mark]
    elif option == options[3]:
        expermen_names = ["eliminated_data_matrix_with_tashkeel"+encoding_mark]
    elif option == options[4]:
        expermen_names = ["full_data_matrix_with_tashkeel"+encoding_mark]
else:
    expermen_names = ["eliminated_data_matrix_without_tashkeel"+encoding_mark,
                      "full_data_matrix_without_tashkeel"+encoding_mark,
                      "eliminated_data_matrix_with_tashkeel"+encoding_mark,
                      "full_data_matrix_with_tashkeel"+encoding_mark]
        

print("types of experiments :   ",  expermen_names)
print("encoding function :   " , vectoriz_fun)
print("fraction :     ",fraction)

    
dataset_path = "../data/Almoso3a_Alshe3rya/data/All_ksaied_cleaned.csv"
dataset = pd.read_csv(dataset_path,index_col=0)
max_bayt_len = np.max((dataset.Bayt_Text.apply(pyarabic.araby.strip_tashkeel).apply(len)))
print("load_dataset Done")
# =============================================================================


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
for file_name in expermen_names:
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
                           
                        runner(dataset,
                               vectoriz_fun,
                               max_bayt_len,
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
                               path_,
                               MULTI_GPU_FLAG,
                               use_CPU)

                        #update experiment_state to done 
                        helpers.update_log_file(Experiement_Name,"done@0", False)
                        
                    print(Experiement_Name,"   Done !!")
                    print("=" * 80)

if len(sys.argv) == 2 and sys.argv[1] == '--test':

    print('End Testing '*20)
    os.system('clear')
    print(comparing_.check_results())
    removeTestingFiles()
