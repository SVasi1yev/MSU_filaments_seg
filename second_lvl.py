import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

import time


class CNN(nn.Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.n_channels = n_channels

        self.conv01 = nn.Conv1d(
            in_channels=self.n_channels, out_channels=3,
            kernel_size=5, stride=3
        )
        self.act01 = nn.ReLU()
        self.conv02 = nn.Conv1d(
            in_channels=3, out_channels=1,
            kernel_size=3, stride=1
        )

        self.fc01 = nn.Linear(in_features=1000, out_features=256)
        self.act02 = nn.ReLU()
        self.fc02 = nn.Linear(in_features=256, out_features=32)

    def forward(self, x):
        x = self.conv01(x)
        x = self.act01(x)
        x = self.conv02(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc01(x)
        x = self.act02(x)
        x = self.fc02(x)
        out = F.sigmoid(x)

        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.gru = nn.GRU(
            input_size=4, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.act01 = nn.ReLU()
        self.fc01 = nn.Linear(in_features=128, out_features=32)
        self.act02 = nn.ReLU()
        self.fc02 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.gru(x)
        x = self.act01(x)
        x = self.fc01(x)
        x = self.act02(x)
        x = self.fc02(x)

        return x


def train(train_dataloader, test_dataloader, net, optimizer, criterion, n_epochs):
    for epoch in range(n_epochs):
        start_time = time.time()
        train_losses = []
        for X, Y in train_dataloader:
            X = X.float()
            optimizer.zero_grad()
            preds = net(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_preds, test_Y = [], []
        for X, Y in test_dataloader:
            X = X.float()
            optimizer.zero_grad()
            preds = net(X)
            test_preds.append(preds)
            test_Y.append(Y)
        test_preds = torch.cat(test_preds).cpu().detach().numpy()
        test_Y = torch.cat(test_Y).cpu().detach().numpy()
        test_rocauc = roc_auc_score(test_Y, test_preds)
        end_time = time.time()
        print(
            "Epoch {}, training loss {:.4f}, test ROCAUC {:.4f}, time {:.2f}".format(
                epoch, np.mean(train_losses), test_rocauc,
                end_time - start_time
            )
        )