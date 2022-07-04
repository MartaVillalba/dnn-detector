# DNN-based detectors for noncoherent MIMO communications

Author: Marta Villalba Cantero.
Directors: Ignacio Santamaría Caballero, Diego Cuevas Fernández.

Master in Data Science (UC-UIMP).

## Abstract

Non-coherent wireless communication systems, in which channel estimation is not needed to decode the transmitted signal, have recently gained renewed interest due to their applications in 5G Ultra-Reliable Low-Latency Communications, or URLLC. In non-coherent multi-antenna communication systems (Multiple-Input Multiple-Output or MIMO), the transmitted signals are Grassmann manifold subspaces or points. A problem that comes with the application of this so-called non-coherent Grassmanian constellations is that the computational cost of the detector grows with the number of codewords K=2^(RT), where R is the code transmission rate in bits/s/Hz and T is the channel coherence time. 

The objective of this project is to study and implement deep neural networks (DNNs) based detectors for non-coherent MIMO communications. Two neural network models are presented, on the one hand a multiclass classification model and, on the other hand, a regression model, and their optimal topology and parameters configuration are analyzed. Also, several preprocessing and post-processing methods are proposed and compared in order to improve neural network performance as a non-coherent MIMO communications detector. The development of deep detectors capable of learning from data will also enable their adaptation to other noise (non Gaussian) and channel (non Rayleigh) models in which the optimal detector is extremely complex or simply does not exist.

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
