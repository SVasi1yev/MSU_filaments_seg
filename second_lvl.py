import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

import time


class CNNDataset(Dataset):
    def __init__(self, clusters_ext):
        self.clusters_ext = clusters_ext
        self.ids = sorted(clusters_ext['ID'].unique())
        self.feas = [
            #             'logreg_score',
            #             'boosting_score',
            'rf_score'
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        out = []
        for f in self.feas:
            out.append(self.clusters_ext[self.clusters_ext['ID'] == id_][f])

        X = np.array(out)
        Y = self.clusters_ext[self.clusters_ext['ID'] == id_]['type'].iloc[0]

        return X, Y


class CNN(nn.Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.n_channels = n_channels

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_channels, out_channels=1,
                kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            #             nn.Conv1d(
            #                 in_channels=3, out_channels=3,
            #                 kernel_size=3, stride=1
            #             ),
            #             nn.ReLU(),
            #             nn.MaxPool1d(2)
        )

        self.clf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(248, 2),
            #             nn.ReLU(),
            #             nn.Dropout(0.5),
            #             nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.clf(x)

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
            preds = net(X)
            Y_ = torch.zeros((Y.size(0), 2))
            Y_[:, Y] = 1
            loss = criterion(preds, Y_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_preds, test_Y = [], []
        for X, Y in test_dataloader:
            X = X.float()
            preds = net(X)
            preds = preds.argmax(axis=1)
            test_preds.append(preds.reshape(-1))
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

# cnn = CNN(1)
# w = torch.Tensor([0.15, 0.85])
# criterion = torch.nn.BCELoss(w)
# optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001)
# train(
#     train_dataloader, test_dataloader,
#     cnn, optimizer, criterion, 30
# )
