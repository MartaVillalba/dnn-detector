"""
    Script to train and evaluate three NN detectors based on soft bits with different number of one-dimensional 
    subspaces in Frobenius norm preprocessing:
    - R = 8
    - R = 16
    - R = 32
"""

import sys
sys.path.append('.')

from lib.processing import generate_subspaces, preprocessing
from lib.dataset import dataset_generation
from lib.model import model_definition_soft_bits, model_training
from lib.utils import save_tensorflow_model, save_model_training, load_model
from lib.simulation import mc_simulation
from lib.plot import plot_model_training
import numpy as np
from params import K, T, M

# Dataset generation parameters
train_SNR = 30
n_samples = 200000

# Generate dataset
dataset = dataset_generation(n_samples, train_SNR)

print("First model - R = 8")
filename = 'Frobenius 1D - soft bits - R = 8 - coherence - 512 n'
R = 8
S_norm_8 = generate_subspaces('Unidimensional', R)

# Define and train NN model
DL_model_1 = model_definition_soft_bits(n_neurons = [1024, 1024, 1024], input_shape = (1, R))
X, y = preprocessing(dataset['X'], dataset['y'], n_samples, 'Frobenius 1D', R, S_norm_8, 'Soft bits')
DL_model_1, history = model_training(DL_model_1, X, y, val_split = 0.2, n_epochs = 200, n_batch = 128)
plot_model_training(history, filename, model_type='Soft bits')
save_tensorflow_model(DL_model_1, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm_8, 
                          [512], 'RMSprop', 0.001, 128, filename, model_type='Soft bits')
DL_model_1 = load_model(filename)

print("Second model - R = 16")
filename = 'Frobenius 1D - soft bits - R = 16 - coherence - 512 n'
R = 16
S_norm_16 = generate_subspaces('Unidimensional', R)

# Define and train NN model
DL_model_2 = model_definition_soft_bits(n_neurons = [512], input_shape = (1, R))
X, y = preprocessing(dataset['X'], dataset['y'], n_samples, 'Frobenius 1D', R, S_norm_16, 'Soft bits')
DL_model_2, history = model_training(DL_model_2, X, y, val_split = 0.2, n_epochs = 200, n_batch = 128)
plot_model_training(history, filename, model_type='Soft bits')
save_tensorflow_model(DL_model_2, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm_16, 
                          [512], 'RMSprop', 0.001, 128, filename, model_type='Soft bits')
DL_model_2 = load_model(filename)

print("Third model - R = 32")
filename = 'Frobenius 1D - soft bits - R = 32 - coherence - 512 n'
R = 32
S_norm_32 = generate_subspaces('Unidimensional', R)

# Define and train NN model
DL_model_3 = model_definition_soft_bits(n_neurons = [512], input_shape = (1, R))
X, y = preprocessing(dataset['X'], dataset['y'], n_samples, 'Frobenius 1D', R, S_norm_32, 'Soft bits')
DL_model_3, history = model_training(DL_model_3, X, y, val_split = 0.2, n_epochs = 200, n_batch = 128)
plot_model_training(history, filename, model_type='Soft bits')
save_tensorflow_model(DL_model_3, filename)
save_model_training(history, n_samples, train_SNR, 'Frobenius 1D', R, S_norm_32, 
                          [512], 'RMSprop', 0.001, 128, filename, model_type='Soft bits')
DL_model_3 = load_model(filename)

# Obtain SER-SNR curve
DL_model_list = [DL_model_1, DL_model_2, DL_model_3]
prep_list = ['Frobenius 1D', 'Frobenius 1D', 'Frobenius 1D']
label_prep_list = ['Soft bits', 'Soft bits', 'Soft bits']
S_list = [S_norm_8, S_norm_16, S_norm_32]
R_list = [8, 16, 32]
train_SNR = [30, 30, 30]
test_SNR = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]) # Tx SNR for MC simulation
num_sim = np.array([2e5,2e5,2e5,2e5,2e6,2e6,2e6,3e6,3e6,3e6,3e6,5e6,3e8,3e8,3e8,3e8])/1e2 # number of channel realizations (coherence blocks)

mc_simulation(test_SNR, num_sim, DL_model_list, prep_list, label_prep_list, R_list, S_list, train_SNR,
                 load_ML = True, post_processing = True, non_linear = False)
                
print("Pipeline finished")