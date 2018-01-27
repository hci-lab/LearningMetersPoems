# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 02:28:40 2018

@author: ali
"""
import re
import string

old_data_path="C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//data//english_dataset.csv"
new_data_path='C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//English_foot.csv'
new_dataset= pd.read_csv(new_data_path, encoding = "ISO-8859-1", index_col=0)
old_dataset= pd.read_csv(old_data_path, encoding = "ISO-8859-1", index_col=0)

new_dataset.head()
new_dataset.columns
new_dataset['Meter'].value_counts()

old_dataset.head()
old_dataset.columns
old_dataset["Meter"].value_counts()
old_dataset=old_dataset[2233:]
old_dataset.head()


new_dataset.columns=['Verse', 'Meter']

merged_data=new_dataset.append(old_dataset, ignore_index=True)
merged_data.head()

merged_data["Meter"].value_counts()

pattern1 = r'.*iam.*'
merged_data['Meter']=merged_data['Meter'].str.replace(pattern1, "iambic")

troch_pattern = r'.*tro.*'
merged_data['Meter']=merged_data['Meter'].str.replace(troch_pattern, "trochaic")

anapaest_pattern = r'.*anap.*'
merged_data['Meter']=merged_data['Meter'].str.replace(anapaest_pattern, "anapaestic")

merged_data["Meter"].value_counts()

merged_data['Verse'].head(100)

def cln(s):
    return re.sub(r'[0-9]+', '', str(s))

merged_data['Verse'] = merged_data['Verse'].apply(cln)

merged_data['Verse'] = merged_data['Verse'].map(lambda x: x.lower())


our_alphabets="".join(list(string.ascii_lowercase)+[" ","\'"])

merged_data['Verse']=merged_data['Verse'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(
                                           lambda x: re.sub(r'  *'," ",x))
merged_data['Meter']=merged_data['Meter'].apply(lambda x: re.sub(r'[^'+our_alphabets+']','',str(x))).apply(
                                          lambda x: re.sub(r'  *'," ",x))
merged_data['Verse']=merged_data['Verse'].apply(lambda x: x.strip())

merged_data['Meter']=merged_data['Meter'].apply(lambda x: x.strip())

merged_data['Verse'].map(lambda x: x.split()).map(len).min()

#merged_data['Meter'].replace('', np.nan, inplace=True)
#merged_data.dropna(how='any', inplace=True, axis=0) 
'''
merged_data['Verse'].replace('', np.nan, inplace=True)
merged_data.dropna(subset=['Verse'], inplace=True)
'''


