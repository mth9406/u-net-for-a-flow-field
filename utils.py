import torch
import numpy as np
import os

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_fluid_data(fluid_path= '../cylinder/Data/Re300', lag= 1, test_size= 0.3, valid_size= 0.1):
    files = list(map(lambda x:os.path.join(fluid_path,x), os.listdir(fluid_path)))
    X, Y = [], []
    for i in range(len(files)-lag):
        X.append(files[i:i+lag])
        Y.append(files[i+lag])
    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= test_size)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size= valid_size)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

class FluidDataset(Dataset):
    """
    fluid_path: data src (default= '../cylinder/Data/Re300)
    lag: time lag (default= 1)
    """
    def __init__(self, file_x, file_y):
        super(FluidDataset, self).__init__()
        self.file_x, self.file_y = file_x, file_y  
        
    def __getitem__(self, index):
        x = np.stack([np.load(file) for file in self.file_x[index]])
        x = torch.FloatTensor(x)
        y = np.load(self.file_y[index])
        y = torch.FloatTensor(y)
        return x, y

    def __len__(self):
        return len(self.file_x)

class EarlyStopping(object):

    def __init__(self, 
                patience: int= 10, 
                verbose: bool= False, delta: float= 0,
                path = './'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta # significant change
        self.path = os.path.join(path, 'latest_checkpoint.pth.tar')
        self.best_score = None
        self.early_stop= False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0


    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        
        torch.save(ckpt_dict, self.path) 
        self.val_loss_min = val_loss