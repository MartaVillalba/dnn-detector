# DNN-based detectors for noncoherent MIMO communications

Author: Marta Villalba Cantero.
Directors: Ignacio Santamaría Caballero, Diego Cuevas Fernández.

Master in Data Science (UC-UIMP).

## Abstract

## Repository structure

* **data**:
    * **codebooks**: Constellations to generate datasets and subspaces.
    * **datasets**: Training datasets.
    * **optimal_detector**: Optimal detector SER-SNR curves.
    * **plots**: Folder to save plots generated with the scripts.
    * **preprocessing_subspaces**: Subspaces for principal angles and Frobenius norm preprocessing.
    * **results**: Folder to save results in MAT files.

* **lib**: Modules with all the functions used in the project.
    * `dataset.py`: Generate training data.
    * `model.py`: Define, train and evaluate a neural network model with TensorFlow.
    * `plot.py`: Plot results.
    * `processing.py`: Process and transform data.
    * `simulation.py`: Monte Carlo simulations.
    * `utils.py`: Load and save data.

* **models**: Trained TensorFlow neural networks.

* **scripts**: Scripts to train and evaluate neural networks using the modules in **lib**.
    * `params.py`: Common parameters.
    * `script_01_snr_comparation.py`: Several NN detectors with the same structure but different training SNR.
    * `script_02_variable_snr.py`: NN detector trained with variable SNR samples.
    * `script_03_n_layer_comparation.py`: Several NN detectors with different number of hidden layers.
    * `script_04_n_neurons_comparation.py`: Several NN detectors with different number of neurons in hidden layers.
    * `script_05_preprocessing.py`: Several NN detectors with different preprocessing methods.
    * `script_06_R_comparation.py`: Several NN detectors with different number of one-dimensional subspaces in Frobenius norm preprocessing.
    * `script_07_soft_bits.py`: Several NN detectors based on soft bits with different number of one-dimensional subspaces in Frobenius norm preprocessing.
