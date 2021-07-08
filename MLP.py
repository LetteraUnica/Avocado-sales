from IPython.display import clear_output
from time import time
import os.path

import torch
from torch.nn import functional as F
import numpy as np
import pylab as pl
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pyro
from pyro import distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS
import seaborn as sns

sns.set_theme()


class MLP_mcmc:
    def __init__(self, num_layers=3, hidden_size=64,
                n_classes=2, num_samples=30, num_chains=1, warmup_steps=100):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.params = dict()

        self.samples = None
        self.n_classes = n_classes
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.warmup_steps = warmup_steps
        if n_classes > 2:
            Exception("Currently the model only supports binary classification")


    def compute_logits(self, x):
        for i in range(1, self.num_layers+2):
            w, b = f"w{i}", f"b{i}"
            # If we are not in the first or last layer create a skip connection
            if i > 1 and i < self.num_layers+1:
                x = x + F.relu(x@self.params[w] + self.params[b])
            else:
                x = F.silu(x@self.params[w] + self.params[b])

        return x

    def model(self, x):
        in_features = x.size()[1]
        out_features = self.hidden_size

        for i in range(1, self.num_layers+2):
            if i == self.num_layers + 1:
                out_features = 1

            w = f"w{i}"
            self.params[w] = pyro.sample(w,
                                dist.Normal(torch.zeros((in_features, out_features)),
                                            torch.ones((in_features, out_features))))

            b = f"b{i}"
            self.params[b] = pyro.sample(b,
                                dist.Normal(torch.zeros(out_features),
                                            torch.ones(out_features)))
            in_features = out_features

        x = pyro.sample("y", dist.Bernoulli(logits=self.compute_logits(x)))

        return x.float()

        
    def fit(self, X, y):
        posterior = pyro.condition(self.model, data={"y": y})

        hmc_kernel = NUTS(posterior, jit_compile=True)

        self.mcmc = MCMC(hmc_kernel, num_samples=self.num_samples,
                         num_chains=self.num_chains,
                         warmup_steps=self.warmup_steps)
        self.mcmc.run(x=X)
        self.samples = self.mcmc.get_samples()
        
        return self

    def predict(self, x, threshold=0.5):
        """Predicts the class a sample will be assigned to"""
        return (self.predict_proba(x) > threshold).int()
    
    def predict_proba(self, x):
        """
        Computes the predictions on x of all the models in samples
        """
        n_models = self.num_samples * self.num_chains
        predictions = torch.zeros(x.size()[0])

        for i in range(n_models):
            predictions += torch.squeeze(torch.sigmoid(self.compute_logits(x)))

        return predictions / n_models
    
    def score(self, X, y):
        if self.samples is None:
            Exception("The fit() method must be called before calling score()")
            
        y_pred = self.predict(X)
        M = confusion_matrix(y, y_pred)
        supports = np.sum(M, axis=1)
        accuracies = np.diag(M) / supports

        print("General Accuracy: ", supports@accuracies / sum(supports))
        for i in range(len(accuracies)):
            print(f"Class {i}: accuracy {accuracies[i]}, support: {supports[i]}")
            
    def summary(self):
        self.mcmc.summary()


from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm



class MLP(nn.Module):
    def __init__(self, input_size, num_layers=3, hidden_size=64, n_classes=2, activation=None, device=None):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.input_size = input_size

        if activation is None:
            self.activation = nn.SiLU
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # architecture
        in_features, out_features = input_size, hidden_size
        layers = []
        for i in range(num_layers+2):
            if i == hidden_size - 1:
                out_features = n_classes - n_classes < 2
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation())
            in_features = out_features

        self.model = nn.Sequential(*layers)
    
        self.name = "deterministic_network"
        self.to(self.device)

    def forward(self, x):
        """ Compute predictions on `inputs`. """
        return self.model(x)

    def train_model(self, train_loader, optimizer, criterion, n_epochs=20, scheduler=None):
        """ Train the network"""
        
        self.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.
            n = 0
            start = time()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.forward(x_batch)
                optimizer.zero_grad() 
                loss = criterion(outputs, y_batch) 
                loss.backward() 
                optimizer.step() 

                epoch_loss += loss.detach().item()
                n += x_batch.size()[0]
            
            if scheduler is not None:
                scheduler.step()

            print(f"Epoch {epoch + 1}\t loss: {epoch_loss/n}\t time: {start-time()}")
            clear_output(wait=True)

    def load(self, fname):
        """Loads (pre-trained) model"""
        self.load_state_dict(torch.load(fname))
        print("Successfully loaded model!")
        
    def save(self, fname, overwrite=False):
        """Saves (trained) model"""
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        torch.save(self.state_dict(), fname)
        print("Successfully saved model!")
      
    def predict(self, dataloader):
        """ Evaluate network on test set. """
        with torch.no_grad():
            correct_predictions = 0.
            n = 0
            # compute predictions on mini-batch
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.forward(x_batch)
                predictions = outputs.argmax(dim=-1)
                correct_predictions += (predictions == y_batch).sum()
                n += x_batch.size()[0]

        return correct_predictions / n


# class MLP_svi(PyroModule):
#     def __init__(self, num_layers=3, hidden_size=64, n_classes=2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.params = dict()

#         self.samples = None
#         self.n_classes = n_classes
#         if n_classes > 2:
#             Exception("Currently the model only supports binary classification")

#         PyroModule[nn.Linear]

#     def init_params(self)

#     def compute_logits(self, x):
#         for i in range(1, self.num_layers+2):
#             w, b = f"w{i}", f"b{i}"
#             # If we are not in the first or last layer create a skip connection
#             if i > 1 and i < self.num_layers+1:
#                 x = x + F.relu(x@self.params[w] + self.params[b])
#             else:
#                 x = F.silu(x@self.params[w] + self.params[b])

#         return x

#     def model(self, x):
#         in_features = x.size()[1]
#         out_features = self.hidden_size

#         for i in range(1, self.num_layers+2):
#             if i == self.num_layers + 1:
#                 out_features = 1

#             w = f"w{i}"
#             self.params[w] = pyro.sample(w,
#                                 dist.Normal(torch.zeros((in_features, out_features)),
#                                             torch.ones((in_features, out_features))))

#             b = f"b{i}"
#             self.params[b] = pyro.sample(b,
#                                 dist.Normal(torch.zeros(out_features),
#                                             torch.ones(out_features)))
#             in_features = out_features

#         x = pyro.sample("y", dist.Bernoulli(logits=self.compute_logits(x)))

#         return x.float()

        
#     def fit(self, X, y):
#         posterior = pyro.condition(self.model, data={"y": y})

#         hmc_kernel = NUTS(posterior, jit_compile=True)

#         self.mcmc = MCMC(hmc_kernel, num_samples=self.num_samples,
#                          num_chains=self.num_chains,
#                          warmup_steps=self.warmup_steps)
#         self.mcmc.run(x=X)
#         self.samples = self.mcmc.get_samples()
        
#         return self

#     def predict(self, x, threshold=0.5):
#         """Predicts the class a sample will be assigned to"""
#         return (self.predict_proba(x) > threshold).int()
    
#     def predict_proba(self, x):
#         """
#         Computes the predictions on x of all the models in samples
#         """
#         n_models = self.num_samples * self.num_chains
#         predictions = torch.zeros(x.size()[0])

#         for i in range(n_models):
#             predictions += torch.squeeze(torch.sigmoid(self.compute_logits(x)))

#         return predictions / n_models
    
#     def score(self, X, y):
#         if self.samples is None:
#             Exception("The fit() method must be called before calling score()")
            
#         y_pred = self.predict(X)
#         M = confusion_matrix(y, y_pred)
#         supports = np.sum(M, axis=1)
#         accuracies = np.diag(M) / supports

#         print("General Accuracy: ", supports@accuracies / sum(supports))
#         for i in range(len(accuracies)):
#             print(f"Class {i}: accuracy {accuracies[i]}, support: {supports[i]}")
            
#     def summary(self):
#         self.mcmc.summary()