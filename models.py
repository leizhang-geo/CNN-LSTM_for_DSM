# coding=utf-8

import torch
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import config as cfg


class ConvNet(nn.Module):
    def __init__(self, num_channels):
        super(ConvNet, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = x[:, :, :, :]
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x).reshape(-1)
        return x

    def num_flat_features(self, x):
        sizes = x.size()[1:]
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.25):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_1 = nn.Linear(hidden_size, 16)
        self.fc_2 = nn.Linear(16, 1)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = torch.tanh(self.fc_1(x[:, -1, :]))
        x = self.fc_2(x).reshape(-1)
        return x

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)


class CovLSTM(nn.Module):
    def __init__(self, cnn_num_channels,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers=1, lstm_dropout=0):
        super(CovLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=cnn_num_channels, out_channels=6, kernel_size=(2, 2), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(2, 2), stride=1, padding=1)
        self.fc_cnn_1 = nn.Linear(64, 16)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        self.fc_lstm_1 = nn.Linear(lstm_hidden_size, 16)

        self.fc_final = nn.Linear(16+16, 1)

    def forward(self, x_cnn, x_lstm):
        x_cnn = x_cnn[:, :, :, :]
        x_cnn = F.max_pool2d(F.relu(self.conv1(x_cnn)), (2, 2))
        x_cnn = F.max_pool2d(F.relu(self.conv2(x_cnn)), (2, 2))
        x_cnn = x_cnn.view(-1, self.num_flat_features(x_cnn))
        x_cnn = F.relu(self.fc_cnn_1(x_cnn))
        # print(x_cnn.shape)

        x_lstm, (h_n, c_n) = self.lstm(x_lstm)
        x_lstm = torch.tanh(self.fc_lstm_1(x_lstm[:, -1, :]))
        # print(x_lstm.shape)

        v_combined = torch.cat((x_cnn, x_lstm), 1)
        pred = self.fc_final(v_combined).reshape(-1)
        return pred

    def num_flat_features(self, x):
        sizes = x.size()[1:]
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features
