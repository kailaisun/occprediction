import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import io
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import datetime
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout):
        super().__init__()
        d_model = 512
        self.pre = nn.Linear(input_dim, d_model)
        self.pre2 = nn.Linear(d_model, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,batch_first=True, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.transformer=nn.Transformer(d_model=d_model,batch_first=True, nhead=n_heads)
        self.fc = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.last=nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, x):
        # b,s,d=x.shape
        # x=x.view(b*s,d)
        x = self.relu(self.pre(x))
        # x=self.relu(self.pre2(x))

        x = self.transformer_encoder(x)
        # x=self.transformer(x,x)
        x = x[:,-1,:]#torch.mean(x, dim=1)  # 对序列维度求平均
        x = self.relu(self.fc(x))
        # x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        x=self.last(x)
        return x
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
