"""Dataset Module for Splitting and Ecnoding"""
import torch.utils.data.dataset as Dataset


class POEMS(Dataset):
    """Dataset Handler;
        * Splits the dataset into (train, test).
        * Performs encoding.

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
        
        # Making those available for the rest methods.
        self.len_maximum_bayt = np.max(dataDF['bayt'].apply(strip_tashkeel).apply(len))
        self.transform = transform
        self.train     = train
        
        
        # Getting Unique Bohor
        classic_bohor = dataDF.bahr.unique()
        
        
        # Splitting the dataDF
        self.split_dataDF(dataDF,
                          self.train_csv_file,  # Outputs csv for training.
                          self.test_csv_file)   # Outputs csv for testing.
        
        
        if  self.train:
            self.train_data = pd.read_csv(self.train_csv_file).values
            print(len(self.train_data))
        else:
            self.test_data = pd.read_csv(self.test_csv_file).values
            print(len(self.test_data))


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



    def __getitem__(self):
        """
        Args:
            Index: starts at 0
            
        Return:
            list: (Bayt, Label)
        """
        # Picking an item
        if self.train:
            item = self.train_data[index]
        else:
            item = self.test_data[index]
    
        # Encoding an item
        if self.transform:
            item = [self.transform(item[0], self.len_maximum_bayt), item[1]]
        
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
