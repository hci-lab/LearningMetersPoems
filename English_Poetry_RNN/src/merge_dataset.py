# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 02:28:40 2018

@author: ali
"""
import re
import string
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import resample
#path of old data
old_data_path="C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//data//english_dataset.csv"
#path of new data
new_data_path='C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//English_foot.csv'
#read new dataset
new_dataset= pd.read_csv(new_data_path, encoding = "ISO-8859-1", index_col=0)
#read old dataset
old_dataset= pd.read_csv(old_data_path, encoding = "ISO-8859-1", index_col=0)
#change coulmns nameof new dataset
new_dataset.columns=['Verse', 'Meter']
'''
new_dataset.head()
new_dataset.columns
'''

#statistics of new dataset
new_dataset['Meter'].value_counts()

'''
old_dataset.head()
old_dataset.columns
'''
#statistics of old dataset
old_dataset["Meter"].value_counts()
#use the cleanest part
old_dataset=old_dataset[2233:]

#merge the new and old dataset
merged_data=new_dataset.append(old_dataset, ignore_index=True)
#statistics of merged dataset
merged_data["Meter"].value_counts()

#rename meters with the same name
pattern1 = r'.*iam.*'
merged_data['Meter']=merged_data['Meter'].str.replace(pattern1, "iambic")

troch_pattern = r'.*tro.*'
merged_data['Meter']=merged_data['Meter'].str.replace(troch_pattern, "trochaic")

anapaest_pattern = r'.*anap.*'
merged_data['Meter']=merged_data['Meter'].str.replace(anapaest_pattern, "anapaestic")

dac_pattern = r'.*dac.*'
merged_data['Meter']=merged_data['Meter'].str.replace(dac_pattern, "dactyl")

merged_data["Meter"].value_counts()

merged_data['Verse'].head(100)

#delete numbers 
def remove_number(verse):
    return re.sub(r'[0-9]+', '', str(verse))

merged_data['Verse'] = merged_data['Verse'].apply(remove_number)

#lowercase all verses
merged_data['Verse'] = merged_data['Verse'].map(lambda x: x.lower())

#all english alphabets
our_alphabets="".join(list(string.ascii_lowercase)+[" ","\'"])
#remove strange characters
merged_data['Verse']=merged_data['Verse'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(
                                           lambda x: re.sub(r'  *'," ",x))
#delete white spaces
merged_data['Verse']=merged_data['Verse'].apply(lambda x: x.strip())

merged_data["Meter"].value_counts()
#remove duplicates (if any)
merged_data=merged_data.drop_duplicates()

#zee=merged_data[len(merged_data['Verse'])<11].index
merged_data['char_count']=merged_data['Meter'].apply(lambda x :len(str(x)))
merged_data['char_count'].min()
merged_data=merged_data.drop(merged_data.index[merged_data['char_count'] <= 10].tolist())

zee=merged_data.index[merged_data['char_count'] <= 3].tolist()
len(zee)
merged_data.loc[zee]
merged_data=merged_data.drop(merged_data.index[merged_data['char_count'] <= 3].tolist())

merged_data["Meter"].value_counts()

#without learning , accuracy will be more than 93%  :"D
merged_data["Meter"].value_counts(normalize=True)


merged_data.to_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//merged_data.csv', encoding = "utf-8")

#shuffled_data = shuffle(merged_data)


#Down-sample Majority Class
# Separate majority and minority classes
df_majority = merged_data[merged_data.Meter =="iambic"]
df_minority = merged_data[merged_data.Meter !="iambic"]
 
# Downsample majority"iambic" class to 5550
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=5550,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.Meter.value_counts()


df_downsampled.to_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//downsampled_iambic.csv', encoding = "utf-8")
