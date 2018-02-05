#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
import os,errno
os.chdir("m://Learning/Master/CombinedWorkspace/Python/DeepLearningMaster/MasterCode/ArabicPoetry-1/Arabic_Poetry_RNN/src/")
# =============================================================================

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint,TensorBoard#,TimeDistributed
#from keras.models import load_model
import preprossesor
import RNN_Model_Helper
import arabic
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
import random as rn
from tensorflow import set_random_seed

print("Imports Done")

# =============================================================================
np.random.seed(7)
set_random_seed(2)
rn.seed(7)
arabic_alphabet = arabic.alphabet
numberOfUniqueChars = len(arabic_alphabet)

# =======================Program Parameters====================================
Experiement_Name = 'Experiement_7_weighted_Loss'
layers_type = 'Bidirectional_LSTM'
num_layers_hidden = 3
weighted_loss_flag = 1
load_weights_flag = 1
with_tashkeel_flag = 0
test_size_param=0.1
validation_split_param = 0.1
n_units = 50
epochs_param = 20
batch_size_param = 512
input_data_path = "../data/All_Data_cleaned.csv"
activation_output_function = 'softmax'
# 0-> last wait | 1 max val_acc
last_or_max_val_acc = 0
#input_data_path = "./data/Almoso3a_Alshe3rya/cleaned_data/All_clean_data.csv"
old_date_flag = 1
new_encoding_flag = 1
earlystopping_patience=-1  
old_data_col =  [0,2,3,5]
new_data_col =  [0,1,2,3,4,6,7]
checkpoints_path ="../checkpoints/"+Experiement_Name+"/"
check_points_file_path = checkpoints_path+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
board_log_dir="../logs/"+Experiement_Name+"/"#+.format(time())

#===============================Concatinated Variables ========================


try:
    os.makedirs(board_log_dir)
    os.makedirs(checkpoints_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Can't create file for checkpoints or for logs please check ")
        raise
print("Input Parameters Defined and Experiement directory created")

# =========================Data Loading========================================
Bayt_Text_Encoded_Stacked, Bayt_Bahr_encoded,max_Bayt_length, label_encoder_output, classes_freq = preprossesor.get_input_encoded_date(input_data_path,old_data_col,with_tashkeel_flag)


# =============================================================================

#==========================Data Spliting=======================================
#need to confirm the train/test distribution is the same ****
X_train, X_test, Y_train, Y_test=train_test_split(Bayt_Text_Encoded_Stacked, #bayts
                                                    Bayt_Bahr_encoded, #classes
                                                    test_size=test_size_param, 
                                                    random_state=0)

#default padding need to check the paramters details
print("Input Train/Test Split done.")

# =========================Model Layers Preparations ==========================
# create model
#K.set_learning_phase(1) #set learning phase
#


model = RNN_Model_Helper.get_model(num_layers_hidden,layers_type,n_units,max_Bayt_length,activation_output_function,load_weights_flag,checkpoints_path,last_or_max_val_acc,label_encoder_output,classes_freq,weighted_loss_flag)


#===========================last_epoch_saver====================================
class last_epoch_saver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #save last epoch weghits 
        self.model.save(checkpoints_path+"weights-improvement-last-epoch.hdf5")
        print("Save last epoch Done! ....")


checkpoint = ModelCheckpoint(check_points_file_path, 
                             monitor='val_acc', 
                             verbose=1,
                             save_best_only=True, 
                             mode='max')

print("Model checkpoint defined to track val_acc")
#===========================tensorboard========================================
tensorboard  = keras.callbacks.TensorBoard(log_dir=board_log_dir , 
                                           histogram_freq=0, 
                                           batch_size=batch_size_param, 
                                           write_graph=True, 
                                           write_grads=True, 
                                           write_images=True, 
                                           embeddings_freq=0, 
                                           embeddings_layer_names=None, 
                                           embeddings_metadata=None)

print("Model tensorboard defined")
#==============================================================================    
#
#===========================earlystopping======================================
earlystopping = keras.callbacks .EarlyStopping(monitor='val_acc',
                                             min_delta=0,
                                             patience=earlystopping_patience,
                                             verbose=1,
                                             mode='auto')


last_epoch_saver_ = last_epoch_saver()

callbacks_list = [checkpoint,tensorboard]

if earlystopping_patience ==-1:
    callbacks_list = [checkpoint,tensorboard,last_epoch_saver_]
    print("Add  checkpoint - tensorboard - last_epoch_saver")
else:
    callbacks_list = [checkpoint,tensorboard,earlystopping,last_epoch_saver_]
    print("Add  checkpoint - tensorboard - earlystopping - last_epoch_saver")
#==============================================================================    
    
print(model.summary())

print("Model Training and validation started")

#=============================Fitting Model====================================
# Fitting the RNN to the Training set
hist = model.fit(X_train, 
                 Y_train, 
                 validation_split = validation_split_param, 
                 epochs=epochs_param, 
                 batch_size=batch_size_param, 
                 callbacks=callbacks_list,
                 verbose=1)
#==============================================================================

#==============================================================================
#save last epoch weghits 
model.save(checkpoints_path+"weights-improvement-last-epoch.hdf5")
print("Save last epoch Done! ....")

#==============================================================================
print("Model Training and validation finished")
#print(history.losses)

#===========================Evaluate model=====================================
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
y_pred = model.predict_classes(X_test)

#print(classification_report(Y_test, y_pred))


#==============================================================================


# ===========================Ploting===========================================
print_model(hist)
# =============================================================================