import torch
import pyro
import numpy as np
import pylab as pl
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch.utils.data import Dataset

def normalize_dataframe(data):
    """Returns a normalized version of the dataframe"""
    x = data.to_numpy()
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)

def bmi(w, h):
    """Computes the bmi"""
    return w/(h**2)

def dataframe_to_tensor(data, normalize=True, dtype=torch.float32):
    """Converts a dataframe into a torch tensor"""
    if normalize: data = normalize_dataframe(data)
    return torch.tensor(data.to_numpy(), dtype=dtype)

def classes_to_one_hot(y, n_classes=2):
    """Returns the hot encoding of the tensor y"""
    y = torch.round(y).int()
    n_obs = y.size()[0]
    y_new = torch.zeros((n_obs, n_classes))
    for i in range(n_obs): 
        y_new[i, y[i]] = 1
    
    return y_new

def one_hot_to_classes(y):
    """Returns the classes from the one hot encoding of y"""
    return torch.argmax(y, dim=1)



class MatrixLoader(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
