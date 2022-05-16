# coding=utf-8
import os

# model hyper-parameters
device = 'cpu'  # 'cpu' or 'cuda'
rand_seed = 314

model_name = 'CNN-LSTM_evi_lsp'  # ['CNN', 'CNN-LSTM_evi', 'CNN-LSTM_lsp', 'CNN-LSTM_evi_lsp', 'LSTM_evi', 'LSTM_lsp', 'LSTM_evi_lsp']

# hyper-parameter of CNN
num_channels = 10

# hyper-parameter of LSTM (small values for parameters for initializing the model training)
lstm_input_size_evi = 1     # feature_size of EVI time series
lstm_input_size_lsp = 11    # feature_size of LSP (phenology) time series
lstm_hidden_size = 8        # hidden size and layers do not need to be large for LSTM
lstm_num_layers = 2
lstm_dropout = 0

# hyper-parameter for training
lr = 1e-3
batch_size = 32
epochs = 1000           # need to consider early stopping to avoid overfitting
eval_interval = 10

data_dir = './data/'
log_dir = './log/'
f_df_samples = os.path.join(data_dir, 'samples_data.csv')   # user need to assign the filename of the sample data (including columns of the target soil property, e.g. soil organic carbon values)
target_var_name = 'soc'     # the column name for the target property (y) that needs to be predicted
f_data_DL_common = os.path.join(data_dir, 'samples_window_common.pkl')     # the pickle file of the input data (X) for CNN (i.e. climate and topographic data with spatially contextual information)
f_data_DL_evi = os.path.join(data_dir, 'samples_ts_evi.pkl')               # the pickle file of the input data (X) for LSTM (i.e. EVI data with temporally dynamic information)
f_data_DL_lsp = os.path.join(data_dir, 'samples_ts_lsp.pkl')               # the pickle file of the input data (X) for LSTM (i.e. phenological data with temporally dynamic information)

train_test_id = 1
f_train_index = os.path.join(data_dir, 'train_test_idx', 'train_{}.pkl'.format(train_test_id))  # the pickle file of the sample id list for the training set
f_test_index = os.path.join(data_dir, 'train_test_idx', 'test_{}.pkl'.format(train_test_id))    # the pickle file of the sample id list for the testing set

model_save_pth = './model/{}_{}.pth'.format(model_name, train_test_id)  # the save path of the model parameters
