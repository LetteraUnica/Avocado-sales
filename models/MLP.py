from IPython.display import clear_output
from time import time
import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.losses = []
        # architecture
        in_features, out_features = input_size, hidden_size
        layers = []
        for i in range(num_layers + 2):
            if i == num_layers + 1:
                out_features = n_classes
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self.activation())
            in_features = out_features

        self.model = nn.Sequential(*layers)
    
        self.name = "deterministic_network"
        self.to(self.device)

    def forward(self, x):
        """ Compute predictions on `inputs`. """
        return torch.squeeze(self.model(x))

    def fit(self, train_loader, optimizer, criterion, n_epochs=20, scheduler=None):
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
                n += 1
            
            if scheduler is not None:
                scheduler.step()
            
            self.losses.append(epoch_loss/n)
            print(f"Epoch: {epoch + 1}\t loss: {epoch_loss/n}\t time: {time()-start}")
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
      
    def accuracy(self, dataloader):
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