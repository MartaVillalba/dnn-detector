"""
    Common parameters.
"""

# NN model names
DL_model_name = ["Frobenius R=8", 
                "Frobenius R=16", 
                "Frobenius R=32"]

# Communication parameters
K = 64
T = 4
M = 2
N = 2
constellation = 'coherence'

# MC simulation block size
max_sim = 1e6

# Paths
codebook_path = "data/codebooks/"
subspaces_path = "data/preprocessing_subspaces/"
dataset_path = "data/datasets/"
plot_path = 'data/plots/'
model_path = 'models/'
results_path = 'data/results/'
ML_path = 'data/optimal_detector/'