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

    
def get_image_paths(path):
    image_paths = []
    for folder in os.listdir(path):
        for fname in os.listdir(os.path.join(path, folder)):
            image_path = os.path.join(folder, fname)
            try:
                if os.path.getsize(os.path.join(path, image_path)) > 20:
                    image_paths.append(image_path)
            except:
                pass

    return image_paths
    
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



# def compute_predictions(f, samples, x):
#     """
#     Computes the predictions on x of all the models in samples
    
#     Returns:
#         A matrix which entry ij represents the the number of times
#         the sample x_i has been assigned to class j
#     """
#     w = samples["w"]
#     b = samples["b"]
#     n_models = w.size()[0]
#     predictions = torch.zeros(x.size()[0])
    
#     for i in range(n_models):
#         predictions += f(x, w, b)
    
#     return predictions / n_models

# def predict_class(samples, x, threshold=0.5):
#     """Predicts the class a sample will be assigned to"""
#     return (compute_predictions(samples, x) > threshold).float()


# def print_class_wise_accuracy(y_true, y_pred):
#     """Prints the general accuracy and the class wise accuracy"""
#     M = confusion_matrix(y_true, y_pred)
#     supports = np.sum(M, axis=1)
#     accuracies = np.diag(M) / supports
    
#     print("General Accuracy: ", supports@accuracies / sum(supports))
#     for i in range(len(accuracies)):
#         print(f"Class {i}: accuracy {accuracies[i]}, support: {supports[i]}")