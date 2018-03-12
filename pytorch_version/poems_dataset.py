"""Dataset Module for Splitting and Ecnoding"""
import torch.utils.data.dataset as Dataset


class POEMS(Dataset):

    def __init__(self):
        pass
    

    def __len__(self):
        pass



    def __getitem__(self):
        pass


    def check_existance(self):
        return os.path.exists(self.train_csv_file) and\
               os.path.exists(self.test_csv_file)
