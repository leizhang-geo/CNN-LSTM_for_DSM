# coding=utf-8

import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm
import models
import data_helper
import config as cfg
import utils


def get_data_loader(x_data, y_data, train_idx, test_idx):
    train_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.Dataset(x_data=x_data, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def get_data_loader_cnnlstm(x_data_cnn, x_data_lstm, y_data, train_idx, test_idx):
    train_dataset = data_helper.DatasetCNNLSTM(x_data_cnn=x_data_cnn, x_data_lstm=x_data_lstm, y_data=y_data, data_index=train_idx, transform=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = data_helper.DatasetCNNLSTM(x_data_cnn=x_data_cnn, x_data_lstm=x_data_lstm, y_data=y_data, data_index=test_idx, transform=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def get_model(model_name, model_save_pth=None):
    if model_name == 'CNN':
        model = models.ConvNet(num_channels=cfg.num_channels)
    elif model_name == 'LSTM_evi':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_evi, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
    elif model_name == 'LSTM_lsp':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_lsp, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
    elif model_name == 'LSTM_evi_lsp':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_evi+cfg.lstm_input_size_lsp, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
    elif model_name == 'CNN-LSTM_evi':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_evi, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
    elif model_name == 'CNN-LSTM_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
    elif model_name == 'CNN-LSTM_evi_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_evi+cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    if model_save_pth is not None:
        model.load_state_dict(torch.load(model_save_pth))
    return model


def predict(model, x_data_cnn=None, x_data_lstm=None):
    model.eval()
    y_pred_list = []

    for i in tqdm(range(len(x_data_cnn))):
        if x_data_cnn is not None:
            x_input_cnn = torch.tensor(x_data_cnn[i:i + 1], dtype=torch.float32)
            if cfg.device == 'cuda':
                x_input_cnn = x_input_cnn.cuda()
        if x_data_lstm is not None:
            x_input_lstm = torch.tensor(x_data_lstm[i:i + 1], dtype=torch.float32)
            if cfg.device == 'cuda':
                x_input_lstm = x_input_lstm.cuda()
        if x_data_cnn is not None and x_data_lstm is None:
            y_pred = model(x_input_cnn)
        elif x_data_cnn is None and x_data_lstm is not None:
            y_pred = model(x_input_lstm)
        elif x_data_cnn is not None and x_data_lstm is not None:
            y_pred = model(x_input_cnn, x_input_lstm)
        else:
            print('Input not valid.')
            sys.exit(0)

        y_pred_list.extend(y_pred.data.cpu().numpy())
    return y_pred_list


def main():
    # Basic setting
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    cfg.device = device
    print('device: {}'.format(device))

    # Load data
    x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y = utils.generate_xy()
    print('x_cnn_common.shape: {}  x_ts_evi.shape: {}  x_ts_lsp.shape: {}  x_ts_evi_lsp.shape: {}\n'.format(x_cnn_common.shape, x_ts_evi.shape, x_ts_lsp.shape, x_ts_evi_lsp.shape))
    train_idx = utils.load_pickle(cfg.f_train_index)
    test_idx = utils.load_pickle(cfg.f_test_index)

    # Build the model
    model_name = 'CNN-LSTM_evi_lsp'
    model = get_model(model_name=model_name, model_save_pth='./model/CNN-LSTM_params.pth')
    if cfg.device == 'cuda':
        model = model.cuda()
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(model_name, model))

    # Predict on test data using the trained model
    print('START PREDICTING\n')
    y_pred_list = predict(model=model, x_data_cnn=x_cnn_common[test_idx], x_data_lstm=x_ts_evi_lsp[test_idx])
    y_true_list = y[test_idx]
    rmse = np.sqrt(metrics.mean_squared_error(y_true_list, y_pred_list))
    mae = metrics.mean_absolute_error(y_true_list, y_pred_list)
    r2 = metrics.r2_score(y_true_list, y_pred_list)

    print('Test_RMSE  = {:.3f}  Test_MAE  = {:.3f}  Test_R2  = {:.3f}'.format(rmse, mae, r2))


if __name__ == '__main__':
    main()
