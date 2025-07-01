import torch
from torch import nn 
from typing import List

class Nn_regression(torch.nn.Module):
    def __init__(self, sizes: List[int], nonLinear: bool = True, epochs = 1000):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if nonLinear and i < len(sizes) - 2:
                layers.append(nn.ReLU())
        
        self.stack = nn.Sequential(*layers)
        self.epochs = epochs
        
    def forward(self, x):
        return self.stack(x)
    
    def fit(self, X_train, y_train):
        EPOCHS = self.epochs
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)
        
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if(epoch % 50 == 0):
                print('epoch: ', epoch)
            
    def predict(self, X_test):
        return self.forward(X_test)
