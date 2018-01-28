# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 22:16:19 2018

@author: ali
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import time

dataset = pd.read_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//All_English_data.csv', encoding = "utf-8",index_col=0)
dataset.head()
dataset.columns
'''
pattern1 = r'.*Metrical foot type: '
pattern2= r'\(.* Metrical foot number: '
pattern3=r"\(.*"

def cln (metre):
    metre.replace(pattern1, "")
    meter= metre.split(' ', 1)[0]
    return meter

def feet(meter):
    meter=meter.split(' ')
    return meter[0]
def feet_number(meter):
    meter=meter.split(' ')
    return meter[1]


dataset['meter'].replace('', np.nan, inplace=True)
dataset['poet'].replace('', np.nan, inplace=True)

dataset.dropna(how='any', inplace=True)
# dataset['Meter']=dataset['Meter'].apply(lambda x: x.strip())
dataset.head(20)

dataset['meter']=dataset['meter'].str.replace(pattern1, "")
dataset['meter']=dataset['meter'].str.replace(pattern2, "")
dataset['meter']=dataset['meter'].str.replace(pattern3, "")
dataset['meter'].head(1000)

dataset['feet']=dataset['meter'].apply(feet)#str.split(' ')[]
dataset['feet'].head(1000)



dataset['feet_number']=dataset['meter'].apply(feet_number)
dataset['feet_number'].head(1000)

'''


'''
dataset['meter']=dataset['meter'].apply(lambda x: x.strip())
dataset['poet']=dataset['poet'].apply(lambda x: x.strip())

dataset['meter'].replace('', np.nan, inplace=True)
dataset['poet'].replace('', np.nan, inplace=True)


dataset.to_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//Separated_features_English_data.csv', encoding = "utf-8")

dataset.groupby('feet').count()


data = pd.read_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//All_English_data.csv', encoding = "utf-8",index_col=0)
'''
data=dataset
data[['foot', 'feet_number']] = data['meter'].str.extract(r'foot type:\s+(?P<foot>\w+).*?foot number:\s+(?P<feet_number>\w+)', expand=True)

data.groupby('foot').count()
'''
data.to_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//Separated_English_data.csv', encoding = "utf-8")
'''
data.columns
data.drop(['meter', 'feet_number'], axis=1, inplace=True)
data.columns

'''
data.drop(['foot_char_count'], axis=1, inplace=True)
data.head()
'''
data.to_csv('C://Users//ali//Documents//GitHub//ArabicPoetry-1//English_Poetry_RNN//English_foot.csv', encoding = "utf-8")

#count characters per foot
#data['foot_char_count'] = [len(str(foot)) for foot in data['foot']]
#data['foot_char_count'].max() 

data['foot'].value_counts()









    







