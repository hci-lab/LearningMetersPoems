#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
import os

#os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/ArabicPoetry
# /ArabicPoetry-1/Arabic_Poetry_RNN/src/")
#os.chdir("/media/mostafaalaa/Main_Hard/Learning/Master/CombinedWorkspace/Python
# /DeepLearningMaster/ArabicPoetry/ArabicPoetry-1/Arabic_Poetry_RNN/src")
# =============================================================================

import numpy as np
import arabic
import random as rn
from tensorflow import set_random_seed
from Exp_Runner import runner
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================

print("Imports Done")

# =============================================================================
np.random.seed(123456)
set_random_seed(123456)
rn.seed(123456)
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =======================Program Parameters====================================

# =============================================================================
layers_type = ["Bidirectional_LSTM","LSTM"]
num_layers_hidden = ["3","6"]
weighted_loss_flag = ["0","1"]
n_units = ["50","82"]
encoded_X_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5"]
encoded_Y_paths = ["../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5",
                   "../data/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5","../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5"]
epochs_param = 50
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

# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 1
activation_output_function = 'softmax'
new_encoding_flag = 1
earlystopping_patience = -1
required_data_col = [0, 2, 3, 5]
test_size_param = 0.1
validation_split_param = 0.1

load_weights_flag = 0

full_classes_encoder_path = "../data/encoders_full_dat.pickle"
eliminated_classes_encoder_path = "../data/encoders_eliminated_data.pickle"

# ===============================Concatinated Variables ========================

counter = 0
for X, Y in zip(encoded_X_paths, encoded_Y_paths):
    file_name = X.split("/")[6].split(".")[0]
    # print(file_name)
    for l_type in layers_type:
        exp_l_type = file_name + "_" + l_type
        # print(exp_l_type)
        for l_num in num_layers_hidden:
            exp_l_num = exp_l_type + "_" + l_num
            for units in n_units:
                exp_unit = exp_l_num + "_" + units
                for w_flag in weighted_loss_flag:
                    exp_w_flag = exp_unit + "_" + w_flag
                    counter += 1
                    Experiement_Name = "Exp_" + str(counter) + "_" + exp_w_flag
                    print(Experiement_Name)
                    runner(X,
                           Y,
                           test_size_param,
                           int(l_num),
                           l_type,
                           validation_split_param,
                           epochs_param,
                           int(units),
                           activation_output_function,
                           load_weights_flag,
                           last_or_max_val_acc,
                           int(w_flag),
                           batch_size_param,
                           earlystopping_patience,
                           Experiement_Name,
                           full_classes_encoder_path,
                           eliminated_classes_encoder_path)
