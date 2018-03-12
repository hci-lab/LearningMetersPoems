"""Dataset Module for Splitting and Ecnoding"""
import torch.utils.data.dataset as Dataset


class POEMS(Dataset):

    def __init__(self):
        pass
    

    def __len__(self):
        pass



    def __getitem__(self):
        pass


    def split_dataDF(self, dataDF, trainFileName, testFileName, test_size=10):
        '''Splits the hole data to train/test.'''
    
        test_DF  = pd.DataFrame(columns=['bayt', 'bahr'])
        train_DF = pd.DataFrame(columns=['bayt', 'bahr'])


        # Raw Labels
        bohor = dataDF.bahr.unique()
        
        for x in bohor:

            bahr_data = dataDF.loc[dataDF['bahr'] == x]
            bahr_size = len(bahr_data)
            testing_number = int(bahr_size * (test_size / 100))

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
                train_DF.to_csv(trainFileName)
                test_DF.to_csv(testFileName)
        except RuntimeError:
            raise RuntimeError('train/test csv are not created!')


    def check_existance(self):
        return os.path.exists(self.train_csv_file) and\
               os.path.exists(self.test_csv_file)
