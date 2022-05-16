# CNN-LSTM_for_DSM
This repository contains the code used for the analysis in the paper: "Zhang et al. A CNN-LSTM model for soil organic carbon content prediction with long time series of MODIS-based phenological variables" (under review).

## Requirement
- Python3
- numpy
- pandas
- scikit-learn
- pytorch
- seaborn
- matplotlib

## Model structure
The CNN-LSTM deep learning model for soil organic carbon (SOC) predictive mapping with inputs of static and dynamic environmental covariates. The spatially contextual features in static variables (e.g. topographic variables) were extracted by CNN, while the temporal features in dynamic variables (e.g. vegetation phenology over a long period of time) were extracted by LSTM. The extracted spatial and temporal features are concatenated to connect fully-connected layers for calculating the outputs (predicted SOC values).

![model_structure](./model_structure.jpg)

## Description of directories and files
- **data (directory)**:
  Here the user needs to put the pickle files of the input data (X and y) for training the CNN-LSTM model. The requried data include:
    - The table file (e.g. csv format) of the sample data (including columns of the target soil property, e.g. soil organic carbon values);
    - The pickle file of input data (X) for CNN (e.g., climate and topographic data with spatially contextual information);
    - The pickle file of input data (X) for LSTM (e.g., EVI data with temporally dynamic information);
    - The pickle file of input data (X) for LSTM (e.g., phenological data with temporally dynamic information).
- **model (directory)**: The folder for storing the model.
- **config.py**: The configuration file for setting the data locations and model hyperparameters.
- **models.py**: The core functions for generating the CNN-LSTM model for the soil prediction.
- **train.py**: It implements data preparation, model initialization and model training procedure.
- **pred.py**: For predicting the target values by using the saved model and evaluating the model performance on the test set.
- **utils.py**: It contains functions for the data loading and generating X and y as the inputs for model training and validating.

## Usage instructions

### Configuration

All model parameters can be set in `config.py`, such as the learning rate, batch size, number of layers, etc.

### Training the model

```python
python train.py
```

The program can save the model parameters in the `model` directory.

### Prediction and Evaluation

```python
python pred.py
```

The saved model can be loaded and evaluating on the test set.

## License

[MIT License](./LICENSE)

## Contact

For questions and supports please contact the author: Lei Zhang 张磊 (zhanglei@smail.nju.edu.cn)

Lei Zhang's [Homepage](https://zlxy9892.github.io/)
