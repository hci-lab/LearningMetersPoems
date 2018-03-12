#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import arabic
import h5py
from helpers import string_with_tashkeel_vectorizer
from pyarabic.araby import strip_tashkeel


def save(nameOfFile, nameOfDataset, dataVar):
    h5f = h5py.File(nameOfFile, 'w')
    h5f.create_dataset(nameOfDataset, data=dataVar)
    h5f.close()



def get_data_matrix(dataset_csv, tashkeel_flag=False):
    """
        dataset_csv: data set file name
        tashkeel: Flase without tashkeel
                  True  with    taskeel
    """

    # 1* Reading data
    data = pd.read_csv(dataset_csv, index_col=0)


    # 2* Getting Unique 16 Bahr
    classic_bohor = data[u'البحر'].unique()[0:16]
    classic_bohor



    # 3*  Filtering data on the 16 Bahr
    is_classic_bahr = data.البحر.isin(classic_bohor)
    filtered_data = data[is_classic_bahr]

    # 4* Getting the longest Bayt
    maximum_bayt_len = np.max((filtered_data[u'البيت'].apply(len)))
    maximum_bayt_len


    # 5* Encoding bohor
    print('Encoding Categories')
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classic_bohor)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    bohor_encoded = onehot_encoder.fit_transform(integer_encoded)
    save('Y_All_Meters', 'Y', bohor_encoded)


    # 6* Encoding Abyat
    print('Encoding Abyat, You may wait :D')
    byot = filtered_data[u'البيت']
    
    if tashkeel_flag is True:
        print("Encoding with tashkeel")
        encoded_bohor = byot.apply(helpers.string_with_tashkeel_vectorizer)
    else:
        print("Removing Tashkeel")
        byot = byot.apply(strip_tashkeel)
        print("Encoding without tashkeel")
        encoded_bohor = byot.apply(helpers.string_with_tashkeel_vectorizer)

    # Dimensions
    # Reshaping the heck!!
    print("Stacking data matrix")
    reshaped_encoded_bohor = np.stack(encoded_bohor, axis=0)
    print('Matrix Dimensions = ' + str(reshaped_encoded_bohor))

    print('Saving Matrix ...')
    if tashkeel_flag is True:
        save('dataMatrix_withTahkeel', 'X', reshaped_encoded_bohor)
    else:
        save('dataMatrix_withoutTahkeel', 'X', reshaped_encoded_bohor)


#print(len(string_with_tashkeel_vectorizer('م أنا')))
