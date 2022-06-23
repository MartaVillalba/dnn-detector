"""
    Script to train and evaluate three NN detectors with different number of hidden layers:
    - 1 hidden layer
    - 3 hidden layers
    - 5 hidden layers
"""

import sys
sys.path.append('.')

from lib.processing import generate_subspaces, preprocessing
from lib.dataset import dataset_generation
from lib.model import model_definition, model_training
from lib.utils import save_tensorflow_model, save_model_training, load_model
from lib.simulation import mc_simulation
from lib.plot import plot_model_training
import numpy as np
from params import K, T, M

# Subspaces for preprocessing
R = 16
S_norm = generate_subspaces('Unidimensional', R)

# Generate dataset
train_SNR = 30
n_samples = 200000
dataset = dataset_generation(n_samples, train_SNR)
X, y = preprocessing(dataset['X'], dataset['y'], n_samples, 'Frobenius 1D', R, S_norm, 'Categorical')

print("First model - 1 hidden layer")
filename = 'Frobenius 1D - R = 16 - coherence - 1 layer'
# Define and train NN model
DL_model_1 = model_definition(n_neurons = [128], input_shape = (1, R))
DL_model_1, history = model_training(DL_model_1, X, y, val_split = 0.2, n_epochs = 1, n_batch = 128)
plot_model_training(history, filename, model_type='Categorical')
save_tensorflow_model(DL_model_1, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm, 
                          [128], 'RMSprop', 0.001, 128, filename, model_type='Categorical')
DL_model_1 = load_model(filename)

print("Second model - 3 hidden layers")
filename = 'Frobenius 1D - R = 16 - coherence - 3 layers'
# Define and train NN model
DL_model_2 = model_definition(n_neurons = [128, 128, 128], input_shape = (1, R))
DL_model_2, history = model_training(DL_model_2, X, y, val_split = 0.2, n_epochs = 1, n_batch = 128)
plot_model_training(history, filename, model_type='Categorical')
save_tensorflow_model(DL_model_2, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm, 
                          [128, 128, 128], 'RMSprop', 0.001, 128, filename, model_type='Categorical')
DL_model_2 = load_model(filename)

print("Third model - 5 hidden layers")
filename = 'Frobenius 1D - R = 16 - coherence - 5 layers'
# Define and train NN model
DL_model_3 = model_definition(n_neurons = [128, 128, 128, 128, 128], input_shape = (1, R))
DL_model_3, history = model_training(DL_model_3, X, y, val_split = 0.2, n_epochs = 1, n_batch = 128)
plot_model_training(history, filename, model_type='Categorical')
save_tensorflow_model(DL_model_3, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm, 
                          [128, 128, 128, 128, 128], 'RMSprop', 0.001, 128, filename, model_type='Categorical')
DL_model_3 = load_model(filename)

# Obtain SER-SNR curve
DL_model_list = [DL_model_1, DL_model_2, DL_model_3]
prep_list = ['Frobenius 1D', 'Frobenius 1D', 'Frobenius 1D']
label_prep_list = ['Categorical', 'Categorical', 'Categorical']
S_list = [S_norm, S_norm, S_norm]
R_list = [16, 16, 16]
train_SNR = [30, 30, 30]
test_SNR = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]) # Tx SNR for MC simulation
num_sim = np.array([2e5,2e5,2e5,2e5,2e6,2e6,2e6,3e6,3e6,3e6,3e6,5e6,3e8,3e8,3e8,3e8])/1e2 # number of channel realizations (coherence blocks)

mc_simulation(test_SNR, num_sim, DL_model_list, prep_list, label_prep_list, R_list, S_list, train_SNR,
                 load_ML = True, post_processing = True, non_linear = False)

print("Pipeline finished")