import torch
import numpy as np
from torch import nn 
from typing import List
from .base_model import BaseModel
from numpy import ndarray

class Nn_regression(torch.nn.Module, BaseModel):
    def __init__(self, sizes: List[int], nonLinear: bool = True, epochs = 1000):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if nonLinear and i < len(sizes) - 2:
                layers.append(nn.ReLU())
        
        self.stack = nn.Sequential(*layers)
        self.epochs = epochs
        
    def forward(self, x: ndarray) -> torch.Tensor:
        x_tensor = torch.from_numpy(x).to(dtype=torch.float32)
        return self.stack(x_tensor)
    
    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        EPOCHS = self.epochs
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)
        
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            
            y_pred = self(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if(epoch % 50 == 0):
                print('epoch: ', epoch)
            
    def predict(self, X_test: ndarray) -> ndarray:
        return self.forward(X_test).detach().numpy()
