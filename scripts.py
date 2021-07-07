import torch
import pyro
import numpy as np
import pylab as pl
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns


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