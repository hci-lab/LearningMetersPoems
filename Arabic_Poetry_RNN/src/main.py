#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
import os,errno
os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/ArabicPoetry/ArabicPoetry-1/Arabic_Poetry_RNN/src/")
# =============================================================================

import numpy as np

#from keras.models import load_model
#from sklearn.metrics import classification_report
import arabic
import random as rn
from tensorflow import set_random_seed
import Exp_Runner
from Exp_Runner import Runner
print("Imports Done")

# =============================================================================
np.random.seed(7)
set_random_seed(2)
rn.seed(7)
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =======================Program Parameters====================================

layers_type = ['Bidirectional_LSTM','LSTM']
num_layers_hidden = [3,6]
weighted_loss_flag = [0,1]
test_size_param=[0.05,0.1]
validation_split_param = [0.02,0.1]

input_data_path = "../data/All_Data_cleaned.csv"
encoded_X_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]

encoded_Y_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5","../data/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]

#input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"
# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 0
activation_output_function = 'softmax'
batch_size_param = 1024
n_units = 50
epochs_param = 50
old_date_flag = 1
new_encoding_flag = 1
earlystopping_patience=-1  
required_data_col =  [0,2,3,5]
load_weights_flag = 0

#new_data_col =  [0,1,2,3,4,6,7]
Experiement_Name = 'Experiement_7_weighted_Loss'
checkpoints_path ="../checkpoints/"+Experiement_Name+"/"
check_points_file_path = checkpoints_path+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
board_log_dir="../logs/"+Experiement_Name+"/"#+.format(time())

#===============================Concatinated Variables ========================



#print(classification_report(Y_test, y_pred))


#==============================================================================


# ===========================Ploting===========================================
print_model(hist)
# =============================================================================