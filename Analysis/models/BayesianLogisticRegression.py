import torch
import pyro
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


class BayesianLogisticRegression:
    def __init__(self, n_classes=2, num_samples=100, num_chains=1, warmup_steps=500):
        self.samples = None
        self.n_classes = n_classes
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.warmup_steps = warmup_steps
        if n_classes > 2:
            Exception("Currently the model only supports binary classification")
    
    def model(self, x):
        """Bayesian logistic regression model"""
        n_obs, n_predictors = x.size()

        w = pyro.sample("w", dist.Normal(torch.zeros(n_predictors),
                                         torch.ones(n_predictors)))
        b = pyro.sample("b", dist.Normal(0, 1))

        logits = x@w + b
        return pyro.sample("y", dist.Bernoulli(logits=logits))
    
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
        w = self.samples["w"]
        b = self.samples["b"]
        n_models = self.num_samples * self.num_chains
        predictions = torch.zeros(x.size()[0])

        for i in range(n_models):
            predictions += torch.sigmoid(x@w[i] + b[i])

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