import os
import torch
import pandas as pd
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader


class Regression(nn.Module):
    def __init__(self, num_features):
        super(Regression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 10)
        self.out = nn.Linear(10, 1)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.out(x)
        return (x)
    

