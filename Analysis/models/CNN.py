from IPython.display import clear_output
from time import time
import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, activation, downsample=False):
        super().__init__()
        layers = []

        for i in range(n_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(activation())
            in_channels = out_channels

        if downsample:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, padding=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self, in_channels, num_layers, num_channels, n_classes=2, activation=None, device=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.n_classes = n_classes

        if activation is None:
            activation = nn.ReLU
        self.activation = activation
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.losses = []
        self.val_accuracies = []

        # architecture
        blocks = []
        for i in range(len(num_layers)):
            out_channels = num_channels[0]
            blocks.append(conv_block(in_channels, out_channels, num_layers[i], activation, True))
            in_channels = out_channels
        
        self.model = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(out_channels, n_classes))
    
        self.name = "deterministic_network"
        self.to(self.device)

    def forward(self, x):
        """ Compute predictions on `inputs`. """
        return self.classifier(torch.mean(self.model(x), dim=[2,3]))

    def fit(self, train_loader, optimizer, criterion, n_epochs=20, scheduler=None, verbose=True, val_loader=None):
        """ Train the network"""
        self.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.
            n = 0
            start = time()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                outputs = self.forward(x_batch)
                optimizer.zero_grad() 
                loss = criterion(outputs, y_batch)
                loss.backward() 
                optimizer.step() 

                epoch_loss += loss.detach().item()
                n += 1
            
            if val_loader is not None:
                val_accuracy = self.accuracy(val_loader).detach().item()
                self.val_accuracies.append(val_accuracy)

            if scheduler is not None:
                scheduler.step()
            
            self.losses.append(epoch_loss/n)

            if verbose:
                message = f"Epoch: {epoch + 1}\t loss: {epoch_loss/n}\t time: {time()-start}"
                if val_loader is not None: message += f"\tval accuracy: {val_accuracy}"
                print(message)
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





class Net(nn.Module):
    def __init__(self, in_channels, l1, l2, n_classes=2, activation=None, device=None):
        super().__init__()

        self.n_classes = n_classes

        if activation is None:
            activation = nn.ReLU
        self.activation = activation
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.losses = []

        # architecture
        self.model = nn.Sequential(nn.Conv2d(in_channels, 6, 5),
                                   activation(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(6, 16, 5),
                                   activation(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Flatten(1),
                                   nn.Linear(16*4*4, l1),
                                   activation(),
                                   nn.Linear(l1, l2),
                                   activation(),
                                   nn.Linear(l2, n_classes)
                                   )
    
        self.name = "deterministic_network"
        self.to(self.device)

    def forward(self, x):
        """ Compute predictions on `inputs`. """
        return self.model(x)

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
