#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
import os#,errno
#os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/ArabicPoetry/ArabicPoetry-1/Arabic_Poetry_RNN/src/")
os.chdir("/media/mostafaalaa/Main_Hard/Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/ArabicPoetry/ArabicPoetry-1/Arabic_Poetry_RNN/src")
# =============================================================================

import numpy as np

#from keras.models import load_model
#from sklearn.metrics import classification_report
import arabic
import random as rn
from tensorflow import set_random_seed
from Exp_Runner import Runner
print("Imports Done")

# =============================================================================
np.random.seed(7)
set_random_seed(2)
rn.seed(7)
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =======================Program Parameters====================================

layers_type = ["Bidirectional_LSTM","LSTM"]
num_layers_hidden = ["3","6"]
weighted_loss_flag = ["0","1"]

input_data_path = "../data/All_Data_cleaned.csv"

encoded_X_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]

encoded_Y_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5","../data/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]

#input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"
# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 0
activation_output_function = 'softmax'
batch_size_param = 2048
n_units = 50
epochs_param = 50
old_date_flag = 1
new_encoding_flag = 1
earlystopping_patience=-1  
required_data_col =  [0,2,3,5]
load_weights_flag = 0
test_size_param= 0.1
validation_split_param = 0.1

#new_data_col =  [0,1,2,3,4,6,7]
#Experiement_Name = 'Experiement_test_full_data'
full_classes_encoder_path = "../data/Almoso3a_Alshe3rya/data/encoders_full_dat.pickle"
eliminated_classes_encoder_path = "../data/Almoso3a_Alshe3rya/data/encoders_eliminated_data.pickle"
#===============================Concatinated Variables ========================



### Need to do some memory management to clear some variables from memory Or
### to reuse it again if possible

counter = 0
for X,Y in zip(encoded_X_paths,encoded_Y_paths):
    file_name = X.split("/")[6].split(".")[0]
    #print(file_name)
    for l_type in layers_type:        
        exp_l_type = file_name + "_" + l_type
        #print(exp_l_type)
        for l_num in num_layers_hidden:
            exp_l_num = exp_l_type + "_" + l_num
            #print(exp_l_num)
            for w_flag in weighted_loss_flag:                
                exp_w_flag = exp_l_num  + "_" + w_flag
                counter += 1
                Experiement_Name = "Exp_" + str(counter) + "_" + exp_w_flag
                print(Experiement_Name )                
                Runner(X,                     
                       Y,
                       test_size_param,
                       int(l_num),
                       l_type,
                       validation_split_param,
                       epochs_param,
                       n_units,         
                       activation_output_function,
                       load_weights_flag,
                       last_or_max_val_acc,
                       int(w_flag),
                       batch_size_param,
                       earlystopping_patience,
                       Experiement_Name,
                       full_classes_encoder_path,
                       eliminated_classes_encoder_path
                       #board_log_dir,
                       #required_data_col,
                       #label_encoder_output,
                       #classes_freq,
                       #max_Bayt_length,
                       #check_points_file_path,
                       #checkpoints_path,           
                       )

      
                    
                    
            
        




#print(classification_report(Y_test, y_pred))


#==============================================================================


# ===========================Ploting===========================================
#print_model(hist)
# =============================================================================