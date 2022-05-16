import os
import numpy as np
import os.path
import torch.utils.data
import utils
import torchvision.transforms as transforms


def default_image_loader(path):
    return utils.load_pickle(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, data_index, transform=None, shuffle=False):
        self.x = x_data[data_index]
        self.y = y_data[data_index]
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_one = self.x[index]
        y_one = self.y[index]
        x_one = torch.tensor(x_one, dtype=torch.float32)
        if self.transform is not None:
            x_one = self.transform(x_one)
        return x_one, y_one

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.y)).tolist())
        else:
            return iter(range(len(self.y)))


class DatasetCNNLSTM(torch.utils.data.Dataset):
    def __init__(self, x_data_cnn, x_data_lstm, y_data, data_index, transform=None, shuffle=False):
        self.x_cnn = x_data_cnn[data_index]
        self.x_lstm = x_data_lstm[data_index]
        self.y = y_data[data_index]
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_one_cnn = self.x_cnn[index]
        x_one_lstm = self.x_lstm[index]
        y_one = self.y[index]
        x_one_cnn = torch.tensor(x_one_cnn, dtype=torch.float32)
        x_one_lstm = torch.tensor(x_one_lstm, dtype=torch.float32)
        if self.transform is not None:
            x_one_cnn = self.transform(x_one_cnn)
        return x_one_cnn, x_one_lstm, y_one

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.y)).tolist())
        else:
            return iter(range(len(self.y)))
