import torch
import sys
import numpy as np
from neuralop.models import FNO
from torch.utils.data import DataLoader, TensorDataset





# adding some code here, will restructure later
class NamedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx].unsqueeze(0)  # Add channel dimension
        y_sample = self.y[idx].permute(2, 0, 1)  # Change from [256, 256, 2] to [2, 256, 256]
#         print(f"Input x shape: {x_sample.shape}, Target y shape: {y_sample.shape}")
        return {"x": x_sample, "y": y_sample}