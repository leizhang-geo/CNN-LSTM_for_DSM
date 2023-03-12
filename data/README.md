Here the user needs to put the pickle files of the input data (X and y) for training the CNN-LSTM model.

The requried data include:
- The table file (e.g. csv format) of the sample data, this file should include columns of the sample location (longitude and latitude) and the value of target soil property, e.g. soil organic carbon values. We recommend that users use their own sample data, or use simulated data for testing (*our sample dataset collected in this study is not publicly available but can be available from the author on reasonable request*).
- The pickle file of input data (X) for CNN (e.g., climate and topographic data with spatially contextual information).
- The pickle file of input data (X) for LSTM (e.g., EVI data with temporally dynamic information).
- The pickle file of input data (X) for LSTM (e.g., phenological data with temporally dynamic information).
