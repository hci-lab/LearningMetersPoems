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

# =============================================================================
import tensorflow as tf
np.random.seed(123456)
tf.reset_default_graph()
with tf.Graph().as_default():
    set_random_seed(123456)
rn.seed(123456)
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)
# =======================Program Parameters====================================

# =============================================================================
layers_type = ["Bidirectional_LSTM" , "LSTM"]
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

#vectoriz_fun = helpers.string_with_tashkeel_vectorizer_OneHot
#vectoriz_fun = helpers.string_with_tashkeel_vectorizer
vectoriz_fun = helpers.two_hot_encoding

expermen_names = ["eliminated_data_matrix_without_tashkeel_two_hot_encoding",
                  "full_data_matrix_without_tashkeel_two_hot_encoding",
                  "eliminated_data_matrix_with_tashkeel_two_hot_encoding",
                  "full_data_matrix_with_tashkeel_two_hot_encoding"]


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
                               MULTI_GPU_FLAG)

                        #update experiment_state to done 
                        helpers.update_log_file(Experiement_Name,"done@0", False)
                        
                    print(Experiement_Name,"   Done !!")
                    print("=" * 80)

if len(sys.argv) == 2 and sys.argv[1] == '--test':

    print('End Testing '*20)
    os.system('clear')
    print(comparing_.check_results())
    removeTestingFiles()
