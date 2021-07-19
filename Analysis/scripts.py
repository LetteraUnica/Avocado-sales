from matplotlib.pyplot import xscale
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
from torchvision.io import read_image
import os


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

    
class load_images(Dataset):
    def __init__(self, path, transform=None):
        self.data_path = path
        self.transform = transform

        self.labels = dict()
        self.image_paths = []
        for i, folder in enumerate(os.listdir(path)):
            self.labels[folder] = i
            for fname in os.listdir(os.path.join(path, folder)):
                image_path = os.path.join(folder, fname)
                try:
                    if os.path.getsize(os.path.join(path, image_path)) > 20:
                        self.image_paths.append(image_path)
                except:
                    pass
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try: 
            image = read_image(os.path.join(self.data_path, image_path)).float()
        except RuntimeError:
            print(image_path)
        label = self.labels[image_path.split("/")[0]]

        if self.transform:
            image = self.transform(image)
   
        return image, torch.tensor(label).long()
    
    def __len__(self):
        return len(self.image_paths)


class avocado_colors():
    def __init__(self):
        self.colors = ["#4a7337", "#ffdb58", "#a44441"]
        self.i = -1

    def __getitem__(self, idx):
        return self.colors[idx]
    
    def __call__(self):
        self.i = (self.i + 1) % len(self.colors)
        return self.colors[self.i]

    def test_colors(self):
        for i in range(len(self.colors)):
            pl.plot(np.random.normal(0,1,1000).cumsum(), color=self.colors[i])

colors = avocado_colors()

def moving_average(x, w=21):
    return np.convolve(x, np.ones(w), 'same') / w

def plot_series(data, ax=pl, legend=None, w=1):
    i = 0
    for column in data:
        if column != "date":
            x = pd.to_datetime(data["date"])
            y = moving_average(data[column], w)
            label = column if legend is None else legend[i]
            ax.plot_date(x, y, "-", label=label, color=colors())
            i += 1
    ax.legend()
    ax.set_xlabel("Date")

def sum_columns(data, columns, name):
    data[name] = data[columns].sum(axis=1)


def params_to_dict(space, bayes_opt_result):
    params_dict = dict()
    for var, value in zip(space, bayes_opt_result.x):
        params_dict[var.name] = value

    return params_dict

def print_optimum(space, bayes_opt_result):
    for var, value in zip(space, bayes_opt_result.x):
        print(var.name, ":", value)

def plot_gp(model, x):
    y_pred, y_std = model.predict(x.reshape(-1, 1), return_std=True)

    y_pred = y_pred.ravel()

    pl.fill_between(x, y_pred - y_std, y_pred + y_std, alpha=0.5, color='k')
    pl.plot(x, y_pred)