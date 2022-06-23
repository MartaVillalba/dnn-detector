"""
    Script to train and evaluate a NN detector trained with variable SNR samples.
"""

import sys
sys.path.append('.')

from lib.processing import generate_subspaces, preprocessing
from lib.dataset import dataset_generation_variable_noise
from lib.model import model_definition, model_training
from lib.utils import save_tensorflow_model, save_model_training, load_model
from lib.simulation import mc_simulation
from lib.plot import plot_model_training
import numpy as np
from params import K, T, M

print("Simple NN - variable SNR training")
# Generate dataset
train_SNR = np.array([0, 10, 20, 30])
n_samples_array = np.array([50000, 50000, 50000, 50000])
n_samples = n_samples_array.sum()
dataset = dataset_generation_variable_noise(n_samples_array, train_SNR)

# Subspaces for preprocessing
filename = 'Frobenius 1D - 2 layers - K = 64 - R = 16 - coherence - SNR 0-30 + concat + normalized'
R = 16
S_norm = generate_subspaces('Unidimensional', R)

# Define and train NN model
DL_model_1 = model_definition(n_neurons = [128, 128], input_shape = (1, R+1))
X, y = preprocessing(dataset['X'], dataset['y'], n_samples, 'Frobenius 1D', R, S_norm, 'Categorical', dataset['SNR_total'])
DL_model_1, history = model_training(DL_model_1, X, y, val_split = 0.2, n_epochs = 100, n_batch = 128)
plot_model_training(history, filename)
save_tensorflow_model(DL_model_1, filename)
save_model_training(history, n_samples_array, train_SNR, 'Frobenius 1D', R, S_norm, 
                         [128, 128], 'RMSprop', 0.001, 128, filename, model_type = 'Categorical')
DL_model_1 = load_model(filename)

# Obtain SER-SNR curve
DL_model_list = [DL_model_1]
prep_list = ['Frobenius 1D']
label_prep_list = ['Categorical']
S_list = [S_norm]
R_list = [R]
test_SNR = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]) # Tx SNR for MC simulation
num_sim = np.array([2e5,2e5,2e5,2e5,2e6,2e6,2e6,3e6,3e6,3e6,3e6,5e6,3e8,3e8,3e8,3e8])/1e2 # number of channel realizations (coherence blocks)

mc_simulation(test_SNR, num_sim, DL_model_list, prep_list, label_prep_list, R_list, S_list, train_SNR,
              clip_rate = None, load_ML = True, post_processing = True, non_linear = False, variable_SNR = True)

print("Pipeline finished")