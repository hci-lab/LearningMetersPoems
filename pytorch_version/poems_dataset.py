"""Dataset Module for Splitting and Ecnoding"""
import torch.utils.data as data
from pyarabic.araby import strip_tashkeel
import pandas as pd
import numpy as np
import torch
import helpers
import os


class POEMS(data.Dataset):
    """Dataset Handler;
        * Splits the dataset into (train, test).
        * Performs encoding.
        * Exports forward and backward dictionaries for classes/encodings.

        * It is used by the DataLoader.
    """

    train_csv_file = 'train_csv_data.csv'
    test_csv_file  = 'test_csv_data.csv'
    test_size      = 10
    def __init__(self, csv_file, train=False, transform=None):
        """
            * Populates train_values or test_values.
            
        Args:
            csv_file (string): On (Bayt & Bahr) Format.
            transform : fucntion pointer, vectorizing method.
            
            
        TODO:
            * Padding length -> From the Hole dataset, before any splitting.
        """
        dataDF = pd.read_csv(csv_file, index_col=0)
        # Getting Unique Bohor
        classes = dataDF.bahr.unique()
        
        # Making those available for the rest methods.
        self.len_maximum_bayt = np.max(dataDF['bayt'].apply(strip_tashkeel).apply(len))
        self.transform = transform
        self.train     = train
        
        
        # dictionary that holds a unique integer for every class
        '''
            * label_encoding_dict: from meter name -> meter number
            * encoding_label_dict: from meter number -> meter name
        '''
        self.label_encoding_dict, self.encoding_label_dict = self.label_encoder(classes)

        '''
        # Testing
        print(self.label_encoding_dict['الخفيف'])
        print(self.encoding_label_dict[2])
        '''
        
        
        # Splitting the dataDF
        self.split_dataDF(dataDF,
                          self.train_csv_file,  # Outputs csv for training.
                          self.test_csv_file)   # Outputs csv for testing.
        
        
        if  self.train:
            self.train_data = pd.read_csv(self.train_csv_file).values
            train_len = len(self.train_data)
            ratio = round(train_len / len(dataDF) *100, 1)
            print('Trains Data: {} obervations, {}% of the hole data.'.format(train_len, ratio))
        else:
            self.test_data = pd.read_csv(self.test_csv_file).values
            test_len = len(self.test_data)
            ratio = round(test_len / len(dataDF) * 100, 1)
            print('Testing Data: {} obervations, {}% of the hole data.'.format(test_len, ratio))


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



    def __getitem__(self, index):
        """
        Args:
            Index: starts at 0
            
        Return:
            list: (Bayt, Label)
        """
        # Picking an item
        if self.train:
            # item: [bayt, bahr_name] *RAW in arabic
            item = self.train_data[index]
        else:
            item = self.test_data[index]
    
        # Encoding an item
        if self.transform:
            encoded_obser = self.transform(item[0], self.len_maximum_bayt)
            label_enc = self.label_encoding_dict[item[1]]
            item = [torch.from_numpy(encoded_obser), label_enc]
        
        return item


    def split_dataDF(self, dataDF, trainFileName, testFileName):
        '''Splits the hole data to train/test.'''
    
        test_DF  = pd.DataFrame(columns=['bayt', 'bahr'])
        train_DF = pd.DataFrame(columns=['bayt', 'bahr'])


        # Raw Labels
        bohor = dataDF.bahr.unique()
        
        for x in bohor:

            bahr_data = dataDF.loc[dataDF['bahr'] == x]
            bahr_size = len(bahr_data)
            testing_number = int(bahr_size * (self.test_size / 100))

            #print(len(bahr_size))
            #print('{} -> {} '.format(bahr_size, int(testing_number)), bahr_size - int(testing_number))

            # take first N.
            test_DF   = test_DF.append(bahr_data[:testing_number])
            bahr_data = bahr_data.drop(bahr_data.index[:testing_number])
            train_DF  = train_DF.append(bahr_data)

        try:
            if self.check_existance():
                print('train/test already exist')
            else:
                train_DF.to_csv(trainFileName, index=False)
                test_DF.to_csv(testFileName, index=False)
        except RuntimeError:
            raise RuntimeError('train/test csv are not created!')


    def check_existance(self):
        return os.path.exists(self.train_csv_file) and\
               os.path.exists(self.test_csv_file)


    def label_encoder(self, classes):
        """
        Args:
            classes [str]: a list of meters names.
        """
        label_encoding_dict = {}
        encoding_label_dict = {}
        label_num = 1

        for class_ in classes:
            label_encoding_dict.update({class_: label_num})
            encoding_label_dict.update({label_num: class_})
            label_num += 1

        return label_encoding_dict, encoding_label_dict

'''
poems_train = POEMS('Diwan_X_Y_Raw_UNIQUE.csv', train=True)
poems_test  = POEMS('Diwan_X_Y_Raw_UNIQUE.csv',
                             train=False,
                             transform=helpers.string_with_tashkeel_vectorizer)

print(len(poems_test.__getitem__(9111)))
print(poems_test.__getitem__(9111)[0])
print(poems_test.__getitem__(9111)[1])

for x in poems_test.label_encoding_dict:
    print('{} {}'.format(x, poems_test.label_encoding_dict[x]))
print('\n\n')
for x in poems_test.encoding_label_dict:
    print('{} {}'.format(x, poems_test.encoding_label_dict[x]))
print('\n\n')
print('\n\n')
for x in poems_train.label_encoding_dict:
    print('{} {}'.format(x, poems_train.label_encoding_dict[x]))
print('\n\n')
for x in poems_train.encoding_label_dict:
    print('{} {}'.format(x, poems_train.encoding_label_dict[x]))
'''
