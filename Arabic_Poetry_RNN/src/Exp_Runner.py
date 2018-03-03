# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:53:53 2018

@author: Mostafa Alaa
"""
#import preprossesor
from preprossesor import get_input_encoded_data_h5
#import RNN_Model_Helper
from RNN_Model_Helper import get_model
import os,errno
import keras
from keras.callbacks import ModelCheckpoint#,TensorBoard#,TimeDistributed
from sklearn.model_selection import train_test_split
import numpy_indexed as npi

def Runner(encoded_X_data_path,
           encoded_Y_data_path,
           test_size_param,
           num_layers_hidden,
           layers_type,
           validation_split_param,
           epochs_param,
           check_points_file_path,
           n_units,
           #max_Bayt_length,
           activation_output_function,
           load_weights_flag,
           checkpoints_path,
           last_or_max_val_acc,
           #label_encoder_output,
           #classes_freq,
           weighted_loss_flag,
           board_log_dir,
           #required_data_col,
           batch_size_param,
           earlystopping_patience):
    try:
        os.makedirs(board_log_dir)
        os.makedirs(checkpoints_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Can't create file for checkpoints or for logs please check ")
            raise
    print("Input Parameters Defined and Experiement directory created")
    
    # =========================Data Loading========================================
    #Bayt_Text_Encoded_Stacked, Bayt_Bahr_encoded,max_Bayt_length, label_encoder_output, classes_freq = preprossesor.get_input_encoded_date(input_data_path,required_data_col,with_tashkeel_flag)
    
    encoded_X_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5"
 
    encoded_Y_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5"
    test_size_param = 0.1
    num_layers_hidden=3
    layers_type='LSTM'
    n_units=50
    last_or_max_val_acc = 0
    weighted_loss_flag=1
    validation_split_param = .01
    
    Bayt_Text_Encoded_Stacked, Bayt_Bahr_encoded = get_input_encoded_data_h5(encoded_X_data_path , encoded_Y_data_path)
    
    # =============================================================================
    
    #==========================Data Spliting=======================================
    print("Start data splitting")
    X_train, X_test, Y_train, Y_test=train_test_split(Bayt_Text_Encoded_Stacked,
                                                        Bayt_Bahr_encoded, #classes
                                                        test_size=test_size_param, 
                                                        random_state=0)
    
    #default padding need to check the paramters details
    print("Input Train/Test Split done.")
    
    # =========================Model Layers Preparations ==========================
    # create model
    unique_classes, classes_freq = npi.count(Bayt_Bahr_encoded, axis=0)
# =============================================================================
#     xxx = hash(tuple(unique_classes))
#     dicts = dict(zip(, classes_freq))
#     
# =============================================================================
    #classes_freq
    #label_encoder_output

    model = get_model(num_layers_hidden,
                      layers_type,
                      n_units,
                      Bayt_Text_Encoded_Stacked.shape[1],
                      Bayt_Text_Encoded_Stacked.shape[2],
                      activation_output_function,
                      load_weights_flag,
                      checkpoints_path,
                      last_or_max_val_acc,
                      weighted_loss_flag,
                      unique_classes, 
                      classes_freq )
    
    
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
    #y_pred = model.predict_classes(X_test)
