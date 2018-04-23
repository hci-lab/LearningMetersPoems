# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:53:53 2018

@author: Mostafa Alaa
"""
import sys
from preprossesor import get_input_encoded_data_h5
from preprossesor import load_encoder
from preprossesor import decode_classes
import numpy as np
from RNN_Model_Helper import get_model
from RNN_Model_Helper import load_weights
from RNN_Model_Helper import wrapped_partial
from RNN_Model_Helper import w_categorical_crossentropy
from RNN_Model_Helper import recall_precision_f1
from helpers import update_log_file
import helpers
import os
import errno
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy_indexed as npi
import pandas as pd
from sklearn.metrics import confusion_matrix
import gc

 

def runner(encoded_x_data_path,
           encoded_y_data_path,
           test_size_param,
           num_layers_hidden,
           layers_type,
           validation_split_param,
           epochs_param,
           n_units,
           activation_output_function,
           load_weights_flag,
           last_or_max_val_acc,
           weighted_loss_flag,
           batch_size_param,
           earlystopping_patience,
           experiment_name,
           full_classes_encoder_path,
           eliminated_classes_encoder_path,
           current_path,
           MULTI_GPU_FLAG):
    # ===============================================================================

    # =============================================================================
    #experiment_name = "Exp_1_eliminated_data_matrix_without_tashkeel_8bitsEncoding_LSTM_3_50_1"
    #encoded_x_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5"
    #encoded_y_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5"
    # =============================================================================

    checkpoints_path = "lib/test_folders/checkpoints/" + experiment_name + "/"
    check_points_file_path = checkpoints_path + "weights-improvement-{epoch:03d}-{val_acc:.5f}.hdf5"
    board_log_dir = "lib/test_folders/logs/" + experiment_name + "/"
    results_dir = "lib/test_folders/Results/" + experiment_name + "/"

    # ===============================================================================
    print('Before ' * 8)
    try:
        os.makedirs(board_log_dir)
        os.makedirs(checkpoints_path)
        os.makedirs(results_dir)
        print('After' * 8)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Can't create file for checkpoints or for logs please check ")
            raise
        print('1' * 100)
        print(sys.exc_info()[0])
    print("Input Parameters Defined and Experiement directory created")

    # ===============================================================================

    # =========================Data Loading========================================

    bayt_text_encoded_stacked, bayt_bahr_encoded = get_input_encoded_data_h5(encoded_x_data_path, encoded_y_data_path)

    # =============================================================================

    # ==============================================================================
    unique_classes, classes_freq = npi.count(bayt_bahr_encoded, axis=0)

    # get saved encoded data

    if "eliminated" not in experiment_name:
        print("working into full data")
        classes_encoder = load_encoder(full_classes_encoder_path)
        names_of_classes = np.apply_along_axis(decode_classes,
                                               0,
                                               unique_classes,
                                               classes_encoder)
        bohor_classes = ['الوافر', 'المنسرح', 'المديد', 'المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل',
                         'الرجز', 'الخفيف', 'البسيط', 'المقتضب', 'الهزج', 'المضارع', 'المتدارك']

    else:
        print("working into eliminated data")
        classes_encoder = load_encoder(eliminated_classes_encoder_path)
        print('Classes Encoder')
        print(type(classes_encoder))
        print(type(unique_classes))
        names_of_classes = np.apply_along_axis(decode_classes, 0, unique_classes, classes_encoder)
        # umar -> elminate el-Madide
        #bohor_classes = ['الوافر', 'المنسرح', 'المديد', 'المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل', 'الرجز', 'الخفيف', 'البسيط']
        bohor_classes = ['الوافر', 'المنسرح', 'المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل', 'الرجز', 'الخفيف', 'البسيط']

    # ==============================================================================

    # Prepare the dataframe
    a = np.stack(names_of_classes, axis=0)
    b = np.stack(classes_freq, axis=0).reshape((1, np.stack(classes_freq, axis=0).shape[0]))
    classes_dest = pd.DataFrame(np.dstack((a, b)).reshape(np.dstack((a, b)).shape[1], np.dstack((a, b)).shape[2]),
                                columns=['Class', 'Cnt'])
    classes_dest['Cnt'] = classes_dest.Cnt.astype(int)

    # ==========================Data Spliting========================================
    print("Start data splitting")
    x_train, x_test, y_train, y_test = train_test_split(bayt_text_encoded_stacked,
                                                        bayt_bahr_encoded,
                                                        test_size=test_size_param,
                                                        random_state=0)

    print("Input Train/Test Split done.")

    # ==============================================================================

    # =========================Model Layers Preparations ===========================
    # create model

    model = get_model(num_layers_hidden,
                      layers_type,
                      n_units,
                      bayt_text_encoded_stacked.shape[1],
                      bayt_text_encoded_stacked.shape[2],
                      activation_output_function,
                      load_weights_flag,
                      checkpoints_path,
                      last_or_max_val_acc,
                      weighted_loss_flag,
                      classes_dest,
                      classes_encoder,
                      MULTI_GPU_FLAG)

    # =============================================================================
    #  umar -> remove that becouse it's redundant
    #model = get_model(int(num_layers_hidden[0]),
    #                            layers_type[0],
    #                            int(n_units[0]),
    #                            bayt_text_encoded_stacked.shape[1],
    #                            bayt_text_encoded_stacked.shape[2],
    #                            activation_output_function,
    #                            load_weights_flag,
    #                            checkpoints_path,
    #                            last_or_max_val_acc,
    #                            int(weighted_loss_flag[0]),
    #                            classes_dest,
    #                            classes_encoder)
    # =============================================================================


    # ===========================lastEpochSaver====================================
    class LastEpochSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # save last epoch weghits
            self.model.save(checkpoints_path + "weights-improvement-last-epoch.hdf5")
            #get expreiment name and update epoch number in log file
            exp_name = checkpoints_path.split('/')[2]
            update_log_file(exp_name,str(epoch),True)
            print("Save last epoch Done! ....")

            helpers.remove_non_max(checkpoints_path)

    checkpoint = ModelCheckpoint(check_points_file_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    print("Model checkpoint defined to track val_acc")
    # ===========================tensorboard=====================================
    tensorboard = keras.callbacks.TensorBoard(log_dir=board_log_dir,
                                              histogram_freq=0,
                                              batch_size=batch_size_param,
                                              write_graph=True,
                                              write_grads=True,
                                              write_images=True,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None,
                                              embeddings_metadata=None)

    print("Model tensorboard defined")
    # ===========================================================================
    #
    # ===========================earlystopping===================================
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  min_delta=0,
                                                  patience=earlystopping_patience,
                                                  verbose=1,
                                                  mode='auto')

    last_epoch_saver_ = LastEpochSaver()

    callbacks_list = [checkpoint, tensorboard]

    if earlystopping_patience == -1:
        callbacks_list = [checkpoint, tensorboard, last_epoch_saver_]
        print("Add  checkpoint - tensorboard - lastEpochSaver")
    else:
        callbacks_list = [checkpoint, tensorboard, earlystopping, last_epoch_saver_]
        print("Add  checkpoint - tensorboard - earlystopping - lastEpochSaver")
    # ===========================================================================

    print(model.summary())

    print("Model Training and validation started")

    print("Clear Variable bayt_text_encoded_stacked from memory")
    del bayt_text_encoded_stacked

    print("Clear Variable del bayt_bahr_encoded from memory")

    del bayt_bahr_encoded

    gc.collect()

    # =============================Fitting Model=================================
    # Fitting the RNN to the Training set
    hist = model.fit(x_train,
                     y_train,
                     validation_split=validation_split_param,
                     epochs=epochs_param,
                     batch_size=batch_size_param,
                     callbacks=callbacks_list,
                     verbose=1)
    # ===========================================================================

    # ===========================================================================
    # save last epoch weghits
    model.save(checkpoints_path + "weights-improvement-last-epoch.hdf5")
    print("Save last epoch Done! ....")

    # ===========================================================================
    print("Model Training and validation finished")
    # print(history.losses)

    print("define partial function w_categorical_crossentropy_pfun")
    sum_of_classes_denesity = classes_dest.Cnt.apply(lambda x: 1 / x).sum()

    w_categorical_crossentropy_pfun = wrapped_partial(w_categorical_crossentropy, classes_dest=classes_dest,
                                                      encoder=classes_encoder,
                                                      sum_of_classes_denesity=sum_of_classes_denesity)
    print("partial function w_categorical_crossentropy_pfun defined")

    # ===========================Evaluate model==================================
    # umar -> make last_or_max_val_acc = 1 to evaluate max
    # Final evaluation of the model
    max_model = load_weights(checkpoints_path, 1, weighted_loss_flag, w_categorical_crossentropy_pfun)

    scores = max_model.evaluate(x_test, y_test, verbose=1)
    print("Exp Results Accuracy : %.2f%%" % (scores[1]))
    print("Exp Results Score : %.2f%%" % (scores[0]))

    predicted = max_model.predict(x_test)
    cmat = confusion_matrix(classes_encoder.inverse_transform(y_test.argmax(1)),
                            classes_encoder.inverse_transform(predicted.argmax(1)), labels=bohor_classes)

    cm_df = pd.DataFrame(cmat, index=bohor_classes, columns=bohor_classes)
    cm_df.to_csv(results_dir + 'CM_MATRIX.csv')
    recall_precision, f1 = recall_precision_f1(cm_df)
    recall_precision.to_csv(results_dir + 'recall_precision_MATRIX.csv')

    # * Add the experiments' results in All_Experiments_Results
    #   which is used to re-conducting the experiment.
    f = open('All_Experiments_Results.csv', 'a')
    f.write('{}, {}\n'.format(experiment_name, scores[1]))
    f.close()

    # ===========================================================================

