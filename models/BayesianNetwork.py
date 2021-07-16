from IPython.display import clear_output
from time import time
import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

import pyro
from pyro import distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive


class BayesianNetwork(PyroModule):
    def __init__(self, det_network):
        super(BayesianNetwork, self).__init__()

        self.det_network = det_network
        self.name = "bayesian network"
        self.device = det_network.device
        self.losses = []
        self.guide = AutoDiagonalNormal(self.model)

    def model(self, X, y=None):
        """ Sets Normal prior distributions and conditions on the observations. """
        if len(X.size()) < 2:
            X = torch.unsqueeze(X, 0)

        priors = dict()
        for key, value in self.det_network.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = dist.Normal(loc=loc, scale=scale)
            priors[str(key)] = prior
    
        lifted_module = pyro.random_module("module", self.det_network, priors)()

        with pyro.plate("data", X.size()[0]):
            out = lifted_module(X)
            obs = pyro.sample("obs", dist.Categorical(logits=out), obs=y)

        return obs
    
    # def guide(self, x_data, y_data=None):
    #     """ Samples from the Variational distribution and returns predictions. """

    #     # take random samples of det_network's weights from the chosen variational family
    #     dists = {}
    #     for key, value in self.det_network.state_dict().items():

    #         # torch.randn_like(x) builds a random tensor whose shape equals x.shape
    #         loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
    #         scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))

    #         # softplus is a smooth approximation to the ReLU function
    #         # which constraints the scale tensor to positive values
    #         distr = dist.Normal(loc=loc, scale=F.softplus(scale))

    #         # add key-value pair to the samples dictionary
    #         dists[str(key)] = distr

    #     # define a random module from the dictionary of distributions
    #     lifted_module = pyro.random_module("module", self.det_network, dists)()

    #     with pyro.plate("data", len(x_data)):
    #         # compute predictions on `x_data`
    #         out = lifted_module(x_data)
    #         preds = F.softmax(out, dim=-1)
        
    #     return preds


    def forward(self, X, n_samples=10, threshold=0.5):
        predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
        preds = predictive(X)["obs"].float()
        return (preds.mean(axis=0) > threshold).int()

    def fit(self, train_loader, optimizer, criterion, n_epochs=20, scheduler=None):
        pyro.clear_param_store()
        self.to(self.device)
        svi = SVI(self.model, self.guide, optimizer, criterion)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.
            n = 0
            start = time()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                epoch_loss += svi.step(x_batch, y_batch)
                n += 1
            
            if scheduler is not None:
                scheduler.step()
            
            self.losses.append(epoch_loss/n)
            print(f"Epoch: {epoch + 1}\t loss: {epoch_loss/n}\t time: {time()-start}")
            clear_output(wait=True)

    def load(self, fname):
        """Loads (pre-trained) model"""
        param_store = pyro.get_param_store()
        param_store.load(fname)
        for key, value in param_store.items():
            param_store.replace_param(key, value.to(self.device), value)
        print("Successfully loaded model!")
        
    def save(self, fname, overwrite=False):
        """Saves (trained) model"""
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        param_store = pyro.get_param_store()
        param_store.save(fname)
        print("Successfully saved model!")
      
    def accuracy(self, dataloader, n_samples=10):
        """ Evaluate network on test set. """
        with torch.no_grad():
            correct_predictions = 0.
            n = 0
            # compute predictions on mini-batch
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                predictions = self.forward(x_batch, n_samples=n_samples)
                correct_predictions += (predictions == y_batch).sum()
                n += x_batch.size()[0]

        return correct_predictions / n