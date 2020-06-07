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
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy_indexed as npi
import pandas as pd
from sklearn.metrics import confusion_matrix
import gc
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pyarabic
import math
 

def runner(dataset,
           vectoriz_fun,
           vectoriz_fun_batch,
           max_bayt_len,
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
           current_path,
           MULTI_GPU_FLAG,
           use_CPU):
    # ===============================================================================

    bayt_text = None
    bayt_bahr_encoded = None
    
    # =============================================================================
    # experiment_name = "Exp_1_eliminated_data_matrix_without_tashkeel_8bitsEncoding_LSTM_3_50_1"
    # encoded_x_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5"
    # encoded_y_data_path = "../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5"
    # =============================================================================
    checkpoints_path = "../Experiements_Info/checkpoints/" + experiment_name + "/"
    check_points_file_path = checkpoints_path + "/weights-improvement-{epoch:03d}-{val_acc:.5f}.hdf5"
    board_log_dir = "../Experiements_Info/logs/" + experiment_name + "/"
    results_dir = "../Experiements_Info/Results/" + experiment_name + "/"
    
    # ===============================================================================
    try:
        os.makedirs(board_log_dir)
        os.makedirs(checkpoints_path)
        os.makedirs(results_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Can't create file for checkpoints or for logs please check ")
            raise
    print("Input Parameters Defined and Experiement directory created")

    # ===============================================================================

    elminated_classic_bohor = ['الوافر', 'المنسرح','المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل', 'الرجز', 'الخفيف', 'البسيط']

    full_bohor_classes = ['الوافر', 'المنسرح', 'المديد', 'المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل', 'الرجز', 'الخفيف', 'البسيط', 'المقتضب', 'الهزج', 'المضارع', 'المتدارك']
    
    # =========================Data Loading========================================

    # encode Bhore full data
    if "eliminated" not in experiment_name:
        print("working into full data")
        filtered_data = dataset.loc[dataset['Bahr'].isin(full_bohor_classes)]
        bohor_classes = full_bohor_classes
    # encode Bhore elminated data
    else:
        print("working into eliminated data")
        filtered_data = dataset.loc[dataset['Bahr'].isin(elminated_classic_bohor)]
        bohor_classes = elminated_classic_bohor

    data_label_encoder = LabelEncoder()
    data_integer_encoded = data_label_encoder.fit_transform(list(filtered_data['Bahr']))
    # binary encode
    data_onehot_encoder = OneHotEncoder(sparse=False)
    data_integer_encoded = data_integer_encoded.reshape(len(data_integer_encoded), 1)
    data_bohor_encoded = data_onehot_encoder.fit_transform(data_integer_encoded)


    # check if with taskeel or not
    if "without_tashkeel" in experiment_name:
        print("work on data without tashkeel")
        bayt_text = filtered_data.Bayt_Text.apply(pyarabic.araby.strip_tashkeel)
    else:
        print("work on data with tashkeel")
        # get text of Abyate array
        bayt_text = filtered_data.Bayt_Text

    bayt_bahr_encoded = data_bohor_encoded
    all_filtered_data = None
    # =============================================================================

    # ==============================================================================
    unique_classes, classes_freq = npi.count(bayt_bahr_encoded, axis=0)

    # get saved encoded data
    classes_encoder = data_label_encoder
    names_of_classes = np.apply_along_axis(decode_classes,
                                           0,
                                           unique_classes,
                                           classes_encoder)
    # ==============================================================================

    # Prepare the dataframe
    a = np.stack(names_of_classes, axis=0)
    b = np.stack(classes_freq, axis=0).reshape((1, np.stack(classes_freq, axis=0).shape[0]))
    classes_dest = pd.DataFrame(np.dstack((a, b)).reshape(np.dstack((a, b)).shape[1], np.dstack((a, b)).shape[2]),
                                columns=['Class', 'Cnt'])
    classes_dest['Cnt'] = classes_dest.Cnt.astype(int)

    # ==========================Data Spliting========================================
    print("Start data splitting")
    x_train, x_test, y_train, y_test = train_test_split(bayt_text,
                                                        bayt_bahr_encoded,
                                                        test_size=test_size_param,
                                                        random_state=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=validation_split_param,
                                                      random_state=0)

    print("Input Train/Test Split done.")

    # ==============================================================================

    # =========================Model Layers Preparations ===========================
    # create model

    one_bayt_text_encoded = vectoriz_fun(bayt_text.iloc[0], max_bayt_len)

    model = get_model(num_layers_hidden,
                      layers_type,
                      n_units,
                      one_bayt_text_encoded.shape[0],
                      one_bayt_text_encoded.shape[1],
                      activation_output_function,
                      load_weights_flag,
                      checkpoints_path,
                      last_or_max_val_acc,
                      weighted_loss_flag,
                      classes_dest,
                      classes_encoder,
                      use_CPU,
                      MULTI_GPU_FLAG)

    # =============================================================================

    
    # ===========================lastEpochSaver====================================
    class LastEpochSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # save last epoch weghits
            self.model.save(checkpoints_path + "weights-improvement-last-epoch.hdf5")
            # get expreiment name and update epoch number in log file
            exp_name = checkpoints_path.split('/')[3]
            state = update_log_file(exp_name,str(epoch+1),True)
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
    del bayt_text


    print("Clear Variable del bayt_bahr_encoded from memory")
    del bayt_bahr_encoded

    gc.collect()
    
    # =============================Batch generator=================================
    
    class ShaarSequence(Sequence):

        def __init__(self, batch_size, bayt_dataset,
                     bhore_dataset,
                     vectorize_fun,
                     max_bayt_len):
            self.batch_size = batch_size
            self.bayt_dataset = bayt_dataset
            self.bhore_dataset = bhore_dataset
            self.start = 0
            self.vectorize_fun = vectorize_fun
            self.max_bayt_len = max_bayt_len

        def __get_batch(self, dataset):
            end = self.start + self.batch_size
            diff = end - len(dataset)
            if diff > 0:
                if diff < self.batch_size:
                    end = len(dataset)
                else:
                    self.start = 0
                    end = self.batch_size
            returned_batch = dataset[self.start:end]
            return returned_batch

        def Bayt_text_batch_generator(self):
            return self.__get_batch(self.bayt_dataset)

        def Bhore_encoded_batch_generator(self):
            return self.__get_batch(self.bhore_dataset)

        def __len__(self):
            return int(np.ceil(len(self.bayt_dataset) / float(self.batch_size)))

        def __getitem__(self, idx):
            x = self.Bayt_text_batch_generator()
            x = self.vectorize_fun(x, self.max_bayt_len)
            y = self.Bhore_encoded_batch_generator()
            self.start += self.batch_size
            return (x, y)

        
    # class generate_one_batch:
    #     def __init__(self,batch_size):
    #         self.batch_size = batch_size
    #         self.start = 0
    #
    #     def get_batch(self):
    #         end = self.start+self.batch_size
    #         diff = end - len(self.dataset)
    #         if diff > 0 :
    #             if diff < self.batch_size :
    #                 end = len(self.dataset)
    #             else:
    #                 self.start = 0
    #                 end = self.batch_size
    #         returned_batch = self.dataset[self.start:end]
    #         self.start +=  self.batch_size
    #         return returned_batch

    # def data_generator(Bayt_text_batch_generator,
    #                    Bhore_encoded_batch_generator,
    #                    vectorize_fun,
    #                    max_bayt_len):
    #     while True:
    #         x = self.get_batch()
    #         x = x.apply(lambda x : vectorize_fun(x , max_bayt_len))
    #         x = np.stack(x,axis=0)
    #         y = Bhore_encoded_batch_generator.get_batch()
    #         yield (x, y)


    # ===========================================================================

    # Bayt_batch_generator = generate_one_batch(batch_size=batch_size_param, dataset=x_train)
    # Bhore_encoded_batch_generator = generate_one_batch(batch_size=batch_size_param, dataset=y_train)
    # steps_per_epoch = math.ceil(len(x_train) / batch_size_param)

    # Bayt_batch_generator_val = generate_one_batch(batch_size=batch_size_param, dataset=x_val)
    # Bhore_encoded_batch_generator_val = generate_one_batch(batch_size=batch_size_param, dataset=y_val)
    # validation_steps = math.ceil(x_val.shape[0] / batch_size_param)

    # =============================Fitting Model=================================
    # hist = model.fit_generator(generator = data_generator(Bayt_batch_generator,
    #                                                       Bhore_encoded_batch_generator,
    #                                                       vectoriz_fun,
    #                                                       max_bayt_len),
    #                            steps_per_epoch=steps_per_epoch,
    #                            validation_data = data_generator(Bayt_batch_generator_val,
    #                                                             Bhore_encoded_batch_generator_val,
    #                                                             vectoriz_fun,
    #                                                             max_bayt_len),
    #                            validation_steps=validation_steps,
    #                            epochs=epochs_param,
    #                            use_multiprocessing=True,
    #                            callbacks=callbacks_list,
    #                            verbose=1)
    # ===========================================================================

    train_seq = ShaarSequence(batch_size_param, x_train, y_train, vectoriz_fun_batch, max_bayt_len)
    val_seq = ShaarSequence(batch_size_param, x_val, y_val, vectoriz_fun_batch, max_bayt_len)


    # =============================Fitting Model=================================
    hist = model.fit_generator(generator=train_seq,
                               # steps_per_epoch=steps_per_epoch,
                               validation_data=val_seq,
                               # validation_steps=validation_steps,
                               epochs=epochs_param,
                               workers=30,
                               max_queue_size=30,
                               use_multiprocessing=True,
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

    w_categorical_crossentropy_pfun = wrapped_partial(w_categorical_crossentropy,
                                                      classes_dest=classes_dest,
                                                      encoder=classes_encoder,
                                                      sum_of_classes_denesity=sum_of_classes_denesity)
    print("partial function w_categorical_crossentropy_pfun defined")

    # ===========================Evaluate model==================================
    # umar -> make last_or_max_val_acc = 1 to evaluate max
    # Final evaluation of the model
    max_model = load_weights(checkpoints_path, 1, weighted_loss_flag, w_categorical_crossentropy_pfun)

    x_test = vectoriz_fun_batch(x_test, max_bayt_len)
    
    scores = max_model.evaluate(x_test ,y_test, verbose=1, batch_size=2048)
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


    # ===========================================================================
    # Clear  memory
    x_train , y_train = None , None
    x_test , y_test = None , None
    x_val , y_val = None , None



    # ===========================================================================
