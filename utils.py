import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, linear_model, ensemble, semi_supervised, datasets, model_selection
import config as cfg


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def calc_dist(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def generate_xy():
    df_samples = pd.read_csv(cfg.f_df_samples)
    y = np.array(df_samples[cfg.target_var_name])
    x_cnn_common = load_pickle(filename=cfg.f_data_DL_common)
    x_ts_evi = load_pickle(filename=cfg.f_data_DL_evi)
    x_ts_lsp = load_pickle(filename=cfg.f_data_DL_lsp)
    x_ts_evi_lsp = np.concatenate([x_ts_evi, x_ts_lsp], axis=2)
    # x_ts_evi = preprocessing.minmax_scale(x_ts_evi.reshape(-1, 1), feature_range=(0, 100), axis=0).reshape(x_ts_evi.shape)
    # y = y[np.random.permutation(len(y))]
    return x_cnn_common, x_ts_evi, x_ts_lsp, x_ts_evi_lsp, y
