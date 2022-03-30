import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time


class CNN(nn.Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.n_channels = n_channels

        self.conv01 = nn.Conv1d(
            in_channels=self.n_channels, out_channels=3,
            kernel_size=9, stride=5
        )
        self.act01 = nn.ReLU()
        self.conv02 = nn.Conv1d(
            in_channels=3, out_channels=1,
            kernel_size=9, stride=5
        )
        self.act02 = nn.ReLU()
        self.conv03 = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=9, stride=5
        )

        self.fc01 = nn.Linear(in_features=1000, out_features=256)
        self.act03 = nn.ReLU()
        self.fc02 = nn.Linear(in_features=256, out_features=32)
        self.act04 = nn.ReLU()
        self.fc03 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.conv01(x)
        x = self.act01(x)
        x = self.conv02(x)
        x = self.act02(x)
        x = self.conv03(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc01(x)
        x = self.act03(x)
        x = self.fc02(x)
        x = self.act04(x)
        x = self.fc03(x)

        return x


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


def train(data_loader, net, optimizer, criterion, n_epochs):
    for epoch in range(n_epochs):
        start_time = time.time()
        train_losses = []
        for X, Y in data_loader:
            optimizer.zero_grad()
            preds = net(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        end_time = time.time()
        print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, np.mean(train_losses),
                                                                   end_time - start_time))