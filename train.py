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


def get_model_and_dataloader(x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y, train_idx, test_idx):
    if cfg.model_name == 'CNN':
        model = models.ConvNet(num_channels=cfg.num_channels)
        train_loader, test_loader = get_data_loader(x_data=x_cnn_common, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'LSTM_evi':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_evi, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_ts_evi, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'LSTM_lsp':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_lsp, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'LSTM_evi_lsp':
        model = models.SimpleLSTM(input_size=cfg.lstm_input_size_evi+cfg.lstm_input_size_lsp, hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_num_layers, dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader(x_data=x_ts_evi_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN-LSTM_evi':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_evi, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_common, x_data_lstm=x_ts_evi, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN-LSTM_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_common, x_data_lstm=x_ts_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    elif cfg.model_name == 'CNN-LSTM_evi_lsp':
        model = models.CovLSTM(cnn_num_channels=cfg.num_channels, lstm_input_size=cfg.lstm_input_size_evi+cfg.lstm_input_size_lsp, lstm_hidden_size=cfg.lstm_hidden_size, lstm_num_layers=cfg.lstm_num_layers, lstm_dropout=cfg.lstm_dropout)
        train_loader, test_loader = get_data_loader_cnnlstm(x_data_cnn=x_cnn_common, x_data_lstm=x_ts_evi_lsp, y_data=y, train_idx=train_idx, test_idx=test_idx)
    else:
        print('Model name is not valid.')
        sys.exit(0)
    return model, train_loader, test_loader


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(model, train_loader, test_loader):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.MSELoss()
    best_rmse, best_mae, best_r2 = np.inf, np.inf, -np.inf
    best_epoch = 1

    for epoch in range(1, cfg.epochs + 1):
        # print('epoch: {}'.format(epoch))
        model.train()
        loss_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                print('input_data_shape:')
                for data in data_input:
                    print(data.shape)
                print()
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            # global_step = batch_idx + (epoch - 1) * int(len(train_loader.dataset) / len(inputs)) + 1
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)
            loss = criterion(y_input, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().data.numpy()
            loss_list.append(loss_val)
        loss_mean = np.mean(loss_list)
        # print('train_loss = {:.3f}'.format(loss_mean))

        if epoch % cfg.eval_interval != 0:
            continue
        print('epoch: {}'.format(epoch))
        model.eval()
        y_input_list = []
        y_pred_list = []
        for batch_idx, data_input in enumerate(train_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)

            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)
        print('Train_RMSE = {:.3f}  Train_MAE = {:.3f}  Train_R2 = {:.3f}'.format(rmse, mae, r2))

        y_input_list = []
        y_pred_list = []
        for batch_idx, data_input in enumerate(test_loader):
            if len(data_input) >= 3:
                x_input_cnn = data_input[0]
                x_input_lstm = data_input[1]
                y_input = data_input[2]
                if cfg.device == 'cuda':
                    x_input_cnn = x_input_cnn.cuda()
                    x_input_lstm = x_input_lstm.cuda()
                    y_input = y_input.cuda()
            else:
                x_input = data_input[0]
                y_input = data_input[1]
                if cfg.device == 'cuda':
                    x_input = x_input.cuda()
                    y_input = y_input.cuda()
            if len(data_input) >= 3:
                x_input_cnn = x_input_cnn.to(cfg.device)
                x_input_lstm = x_input_lstm.to(cfg.device)
                y_pred = model(x_input_cnn, x_input_lstm)
            else:
                x_input = x_input.to(cfg.device)
                y_pred = model(x_input)
            y_pred_list.extend(y_pred.data.cpu().numpy())
            y_input_list.extend(y_input.data.cpu().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(y_input_list, y_pred_list))
        mae = metrics.mean_absolute_error(y_input_list, y_pred_list)
        r2 = metrics.r2_score(y_input_list, y_pred_list)

        torch.save(model.state_dict(), cfg.model_save_pth)
        print('Test_RMSE  = {:.3f}  Test_MAE  = {:.3f}  Test_R2  = {:.3f}'.format(rmse, mae, r2))
        print()


def main():
    # Basic setting
    device = torch.device('cuda:0' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Load data
    x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y = utils.generate_xy()
    print('x_cnn_common.shape: {}  x_ts_evi.shape: {}  x_ts_lsp.shape: {}  x_ts_evi_lsp.shape: {}\n'.format(x_cnn_common.shape, x_ts_evi.shape, x_ts_lsp.shape, x_ts_evi_lsp.shape))
    # sys.exit(0)

    # Build the model
    train_idx = utils.load_pickle(cfg.f_train_index)
    test_idx = utils.load_pickle(cfg.f_test_index)
    model, train_loader, test_loader = get_model_and_dataloader(x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y, train_idx, test_idx)
    if cfg.device == 'cuda':
        model = model.cuda()
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, model))
    
    # Train the model
    input('Press enter to start training...\n')
    print('START TRAINING\n')
    train_model(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
