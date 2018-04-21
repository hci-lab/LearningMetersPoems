
# coding: utf-8

# # Libraries Loading

# In[ ]:


import numpy as np
import pandas as pd
import arabic
import pyarabic
import helpers
from helpers import Clean_data,separate_token_with_dicrites,save_h5
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import array, argmax
import os,errno,re,sys
import pickle
from itertools import product 
from pyarabic.araby import strip_tashkeel, strip_tatweel
import h5py
import random as rn
##########################
rn.seed(123456)
np.random.seed(123456)
input_data_path = "../data/Almoso3a_Alshe3rya/data/All_ksaied.csv"


# In[ ]:


try:
    os.makedirs("../data/Encoded/8bits/WithoutTashkeel/Eliminated/")
    os.makedirs("../data/Encoded/8bits/WithoutTashkeel/Full_Data/")
    os.makedirs("../data/Encoded/8bits/WithTashkeel/Eliminated/")
    os.makedirs("../data/Encoded/8bits/WithTashkeel/Full_Data/")
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Can't create file for checkpoints or for logs please check ")
        raise
print("Paths created")


# # Data Loading

# In[ ]:


input_data = pd.read_csv(input_data_path, index_col=0)
len(input_data)


# In[ ]:


#sample = input_data.sample(1000)
#sample
len(input_data)


# In[ ]:


#input_data = pd.read_csv(input_data_path, index_col=0)

# TAHA
#input_data = sample

input_data.info(memory_usage='deep')
bahr = 'Bahr'
bayt = 'Bayt_Text'
#input_data  = input_data[[bayt, bahr]]
bahr = 'البحر'
bayt = 'البيت'
input_data  = input_data[[bayt, bahr]]
input_data.columns = ['Bayt_Text','Bahr']
our_alphabets = "".join(arabic.alphabet) + "".join(arabic.tashkeel)+" "
our_alphabets = "".join(our_alphabets)
input_data['Bayt_Text'] = input_data['Bayt_Text'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(lambda x: re.sub(r'  *'," ",x)).apply(lambda x: re.sub(r'ّ+', 'ّ', x)).apply(lambda x: x.strip())
#input_data


# In[ ]:


# Taha: first class of the first observation
row_first_observation_class = input_data.iloc[0][1]
row_first_observation_class


# # Data Analysis

# In[ ]:


# Taha: Counts the opservations of each class
input_data.iloc[:, 1].value_counts()


# # Data Filteration

# In[ ]:


elminated_classic_bohor = ['الوافر', 'المنسرح','المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل',
                         'الرجز', 'الخفيف', 'البسيط']
elminated_classic_bohor


# In[ ]:


full_bohor_classes = ['الوافر', 'المنسرح', 'المديد', 'المجتث', 'المتقارب', 'الكامل', 'الطويل', 'السريع', 'الرمل',
                         'الرجز', 'الخفيف', 'البسيط', 'المقتضب', 'الهزج', 'المضارع', 'المتدارك']
full_bohor_classes


# In[ ]:


full_filtered_data = input_data.loc[input_data['Bahr'].isin(full_bohor_classes)]
full_filtered_data['Bayt_Text'] = full_filtered_data['Bayt_Text'].apply(pyarabic.araby.strip_tatweel)
full_filtered_data['Bayt_Text'] = full_filtered_data['Bayt_Text'].apply(helpers.factor_shadda_tanwin)
#full_filtered_data.head()
# Taha
full_filtered_data.iloc[:, 1].value_counts()


# In[ ]:


eliminated_filtered_data = input_data.loc[input_data['Bahr'].isin(elminated_classic_bohor)]
eliminated_filtered_data['Bayt_Text'] = eliminated_filtered_data['Bayt_Text'].apply(pyarabic.araby.strip_tatweel)
eliminated_filtered_data['Bayt_Text'] = eliminated_filtered_data['Bayt_Text'].apply(helpers.factor_shadda_tanwin)
# eliminated_filtered_data.head()

# Taha
eliminated_filtered_data.iloc[:, 1].value_counts()


# In[ ]:


#eliminated_filtered_data.to_csv('Big_eliminated_filtered_data.csv')
#full_filtered_data.to_csv('Big_full_filtered_data.csv')
#eliminated_filtered_data = pd.read_csv('Big_eliminated_filtered_data.csv')
#full_filtered_data = pd.read_csv('Big_full_filtered_data.csv')


# # Encode Classes

# In[ ]:


'''
    Taha; Encoding Full Data classes as OneHot vectors. 
'''

full_data_label_encoder = LabelEncoder()
full_data_integer_encoded = full_data_label_encoder.fit_transform(list(full_filtered_data["Bahr"]))
# binary encode
full_data_onehot_encoder = OneHotEncoder(sparse=False)
full_data_integer_encoded = full_data_integer_encoded.reshape(len(full_data_integer_encoded), 1)
full_data_bohor_encoded = full_data_onehot_encoder.fit_transform(full_data_integer_encoded)
save_h5('../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5', 'Y', full_data_bohor_encoded)
save_h5('../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_Y_Meters.h5', 'Y', full_data_bohor_encoded)
#saving with pickle: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
print('saving preprocessors....')
with open("../encoders_full_dat.pickle", 'wb') as pre_file:
    pickle.dump(full_data_label_encoder, pre_file, protocol=pickle.HIGHEST_PROTOCOL)
print('saved ....')


# In[ ]:


### testing
#print('loading saved encoders from %s'%(encoders_file_name))
with open("../encoders_full_dat.pickle", 'rb') as f:
    encoders = pickle.load(f)

##testing 
#inverted = encoders.inverse_transform([argmax(full_filtered_data["Bahr"][1,])])


# Taha
Ys = h5py.File('../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_Y_Meters.h5', 'r')
class_of_first_observation = Ys['Y'][0]
decoded_first_class = encoders.inverse_transform([argmax( class_of_first_observation   )])

if decoded_first_class == row_first_observation_class:
    print('NICE, Classes Encoded properly')
else:
    print('There is a problem in encoding-decoding classes.')


# In[ ]:


'''
    Taha; This cell is entirly doublicated.
    It Encodes the eliminated data classes.
'''
eliminated_data_label_encoder = LabelEncoder()
eliminated_data_integer_encoded = eliminated_data_label_encoder.fit_transform(list(eliminated_filtered_data["Bahr"]))
# binary encode
eliminated_data_onehot_encoder = OneHotEncoder(sparse=False)
eliminated_data_integer_encoded = eliminated_data_integer_encoded.reshape(len(eliminated_data_integer_encoded), 1)
eliminated_data_bohor_encoded = eliminated_data_onehot_encoder.fit_transform(eliminated_data_integer_encoded)
save_h5('../data/Encoded/8bits/WithoutTashkeel/Eliminated/Eliminated_data_Y_Meters.h5', 'Y', eliminated_data_bohor_encoded)
save_h5('../data/Encoded/8bits/WithTashkeel/Eliminated/Eliminated_data_Y_Meters.h5', 'Y', eliminated_data_bohor_encoded)
#saving with pickle: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
print('saving preprocessors....')
with open("../encoders_eliminated_data.pickle", 'wb') as pre_file:
    pickle.dump(eliminated_data_bohor_encoded, pre_file, protocol=pickle.HIGHEST_PROTOCOL)
print('saved ...')


# # Clean The Data

# ## WITH TASHKEEL

# ### FULL DATA

# In[ ]:


bayt_len_full_withtashkeel = np.max((full_filtered_data.Bayt_Text.apply(separate_token_with_dicrites).apply(len)))
print("bayt_len_full_withtashkeel done..")
full_filtered_data_with_tashkeel_cleaned = Clean_data(data_frame = full_filtered_data,max_bayt_len = bayt_len_full_withtashkeel,verse_column_name = 'Bayt_Text')
print("full_cleaned_data_with_tashkeel cleaned ..")
full_cleaned_data_with_tashkeel_encoded = full_filtered_data_with_tashkeel_cleaned.Bayt_Text.apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, bayt_len_full_withtashkeel))
print("full_cleaned_data_with_tashkeel_encoded encoded ..")
full_cleaned_data_with_tashkeel_encoded_stacked = np.stack(full_cleaned_data_with_tashkeel_encoded,axis=0)
# save('../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5', 'X', full_cleaned_data_with_tashkeel_encoded_stacked) 
# Taha
save_h5('../data/Encoded/8bits/WithTashkeel/Full_Data/full_data_matrix_with_tashkeel_8bitsEncoding.h5', 'X', full_cleaned_data_with_tashkeel_encoded_stacked) 
print("full_cleaned_data_with_tashkeel_encoded saved ..")
print(full_cleaned_data_with_tashkeel_encoded_stacked.shape)

#### ELIMINATED DATA

bayt_len_eliminated_withtashkeel = np.max((eliminated_filtered_data.Bayt_Text.apply(separate_token_with_dicrites).apply(len)))
print("bayt_len_eliminated_withtashkeel done..")
eliminated_filtered_data_with_tashkeel_cleaned = Clean_data(data_frame = eliminated_filtered_data,max_bayt_len = bayt_len_eliminated_withtashkeel  ,verse_column_name = 'Bayt_Text')
print("eliminated_cleaned_data_with_tashkeel cleaned ..")
eliminated_cleaned_data_with_tashkeel_encoded = eliminated_filtered_data_with_tashkeel_cleaned.Bayt_Text.apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, bayt_len_eliminated_withtashkeel))
print("eliminated_cleaned_data_with_tashkeel_encoded encoded ..")
eliminated_cleaned_data_with_tashkeel_encoded_stacked = np.stack(eliminated_cleaned_data_with_tashkeel_encoded,axis=0)
save_h5('../data/Encoded/8bits/WithTashkeel/Eliminated/eliminated_data_matrix_with_tashkeel_8bitsEncoding.h5', 'X', eliminated_cleaned_data_with_tashkeel_encoded_stacked) 
print("full_cleaned_data_with_tashkeel_encoded saved ..")
print(eliminated_cleaned_data_with_tashkeel_encoded_stacked.shape)
#encoded_byot = data_with_Tashkeel_cleaned.apply(lambda x: helpers.string_with_tashkeel_vectorizer_OneHot(x, maximum_bayt_len))

## WITHOUT TASHKEEL

### FULL DATA

# RUN HERE
full_filtered_data_without_tashkeel = full_filtered_data['Bayt_Text'].apply(pyarabic.araby.strip_tashkeel)
bayt_len_full_without_tashkeel = np.max((full_filtered_data_without_tashkeel.apply(len)))
print("bayt_len_full_without_tashkeel done..")
full_cleaned_data_without_tashkeel_encoded = full_filtered_data_without_tashkeel.apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, bayt_len_full_without_tashkeel))
print("full_filtered_data_without_tashkeel encoded ..")
full_filtered_data_without_tashkeel_staked = np.stack(full_cleaned_data_without_tashkeel_encoded,axis=0)
save_h5('../data/Encoded/8bits/WithoutTashkeel/Full_Data/full_data_matrix_without_tashkeel_8bitsEncoding.h5', 'X', full_filtered_data_without_tashkeel_staked) 
print("full_filtered_data_without_tashkeel_staked saved ..")
print(full_filtered_data_without_tashkeel_staked.shape)

### ELIMINATE

eliminated_filtered_data_without_tashkeel = eliminated_filtered_data['Bayt_Text'].apply(pyarabic.araby.strip_tashkeel)
bayt_len_eliminated_without_tashkeel = np.max((eliminated_filtered_data_without_tashkeel.apply(len)))
print("bayt_len_eliminated_without_tashkeel done..")
eliminated_cleaned_data_without_tashkeel_encoded = eliminated_filtered_data_without_tashkeel.apply(lambda x: helpers.string_with_tashkeel_vectorizer(x, bayt_len_eliminated_without_tashkeel))
print("eliminated_filtered_data_without_tashkeel encoded ..")
eliminated_filtered_data_without_tashkeel_staked = np.stack(eliminated_cleaned_data_without_tashkeel_encoded,axis=0)
save_h5('../data/Encoded/8bits/WithoutTashkeel/Eliminated/eliminated_data_matrix_without_tashkeel_8bitsEncoding.h5', 'X', full_filtered_data_without_tashkeel_staked) 
print("eliminated_filtered_data_without_tashkeel_staked saved ..")

print(eliminated_filtered_data_without_tashkeel_staked.shape)


## Congratualtions :) Finished :) :) :)


