"""
    Functions to process and transform data.
"""

from scripts.params import K, T, M, N, constellation, codebook_path, subspaces_path
import numpy as np
import scipy
import tensorflow as tf
import scipy.io as sio
from copy import deepcopy

def preprocessing(X, y, n_samples, type, R = None, S = None, labels_type = 'Categorical', SNR = None):
    """Preprocessing applied to dataset.

    Args:
        X: Input samples.
        y: Output samples.
        n_samples: Number of samples in the dataset.
        type: Type of preprocessing applied to X (Vectorization, Principal angles, 
              Frobenius, Frobenius 1D, Frobenius 1D normalized).
        R: Number of subspaces used for preprocessing.
        S: Subspaces used for preprocessing.
        labels_type: Type of preprocessing applied to y (Categorical, Soft bits).
        SNR: SNR of each samples in variable SNR dataset.

    Returns:
        X, y: Preprocessed samples.
    """

    # Labels
    if labels_type == 'Categorical':
        y_bin = tf.keras.utils.to_categorical(y, K, dtype = np.int8).reshape(n_samples, 1, K)
    elif labels_type == 'Soft bits':
        n_bits = int(np.log2(K))
        y_bin = np.zeros((n_samples, 1, n_bits), dtype=np.int8)
        for i, label in enumerate(y):
            bit_label = np.binary_repr(label, width = n_bits)
            y_bin[i, 0, :] = [int(bit) for bit in bit_label]

    if type == 'Vectorization':
        X_flat = np.empty((n_samples, 1, 2*T*M))
        for i in range(0, n_samples):
            X_flat[i, 0, :] = np.array([np.real(X[i, :, :]), np.imag(X[i, :, :])]).reshape((1, 2*T*M))
        return X_flat, y_bin

    elif type == 'Principal angles':
        X_angles = np.zeros((n_samples, 1, M*R))
        for i in range(n_samples):
            X_bar = scipy.linalg.orth(X[i, :, :])
            angles = np.linalg.svd( X_bar.conj().T@S , compute_uv=False )
            X_angles[i, 0, :] = angles.flatten(order = 'F')
        return X_angles, y_bin

    elif type == 'Frobenius':
        X_norm = np.zeros((n_samples, 1, R))
        for i in range(n_samples):
            X_norm[i, :, :] = np.linalg.norm(X[i, :, :].conj().T@S, axis = (1, 2), ord = 'fro').reshape(1, R)
        return X_norm, y_bin

    elif type == 'Frobenius 1D':
        X_norm = np.zeros((n_samples, 1, R))
        for i in range(n_samples):
             X_norm[i, :, :] = np.linalg.norm(X[i, :, :].conj().T@S, axis = (1, 2), ord = 'fro').reshape(1, R)
        if SNR is not None:
            SNR_norm = SNR.reshape(n_samples, 1, 1)/SNR.max()
            X_norm = np.concatenate((X_norm, SNR_norm), axis = 2)
        return X_norm, y_bin
    
    elif type == 'Frobenius 1D normalized':
        X_norm = np.zeros((n_samples, 1, R))
        for i in range(n_samples):
            norm = np.linalg.norm(X[i, :, :].conj().T@S, axis = (1, 2), ord = 'fro').reshape(1, R)
            X_norm[i, :, :] = norm/norm.sum()
        return X_norm, y_bin


def generate_subspaces(method, R = K):
    """Generate subspaces for preprocessing.

    Args:
        method: Method to generate subspaces (QR, UB, Codebook, Unidimensional).
        R: Number of subspaces to generate.

    Returns:
        S: Generated subspaces.
    """
    if method == 'QR':
        S = np.zeros( (R,T,M) , dtype = complex)
        for i in range(R):
            X = ( np.random.normal(0,1,(T,M)) + 1j*np.random.normal(0,1,(T,M)) )/np.sqrt(2) 
            Q, _ = np.linalg.qr(X)
            S[i, :, :] = Q[:,:M]
    elif method == 'UB':
        filename = "preprocessing_subspaces_" + method + "_R" + str(R) + "_T" + str(T) + "_M" + str(M)
        subspaces = sio.loadmat(subspaces_path + filename + ".mat")
        S_aux = subspaces['pre_subspaces']
        S = np.zeros( (R, T, M), dtype = complex)
        for i in range(R):
            S[i,:,:] = S_aux[:,:,i]
    elif method == 'Codebook':
        filename = "codebook_K" + str(K) + "_T" + str(T) + "_M" + str(M) + "_" + constellation
        dataset_codebook = sio.loadmat(codebook_path + filename + ".mat")
        Codebook = dataset_codebook['C']
        S = np.zeros( (K, T, M), dtype = complex)
        for i in range(K):
            S[i,:,:] = Codebook[:,:,i]
    elif method == 'Unidimensional':
        filename = "preprocessing_subspaces_R" + str(R) + "_T" + str(T) + "_M1"
        subspaces = sio.loadmat(subspaces_path + filename + ".mat")
        S_aux = subspaces['pre_subspaces']
        S = np.zeros( (R, T, 1), dtype = complex)
        for i in range(R):
            S[i,:,0] = S_aux[:,0,i]
    return S


def soft_bits_post_processing(prediction, prediction_bits, Y, true_labels, S_codebook, n_samples):
    """Post-processing based on first and second largest probabilities obtained 
    by regression (soft bits) neural network.

    Args:
        prediction: Output of neural network after evaluation.
        prediction_bits: Output of neural network converted to bits (0 or 1).
        Y: Non-preprocessed samples.
        true_labels: Index of correct codeword of each sample.
        S_codebook: Codebook subspaces.
        n_samples: Number of samples.

    Returns:
        SER: Symbol Error Rate after post-processing.
    """

    # Number of bits
    n_bits = int(np.log2(K))

    # Index of the bit prediction closer to 0.5
    doubtful_bit = np.argmin(np.abs(prediction - 0.5), axis = 2).flatten()

    # Two possible predictions based on doubtful bit
    prediction_1 = deepcopy(prediction_bits)
    prediction_2 = deepcopy(prediction_bits)
    prediction_1[np.arange(int(n_samples)), 0, doubtful_bit] = 0
    prediction_2[np.arange(int(n_samples)), 0, doubtful_bit] = 1

    # Convert binary prediction to integer
    pot_matrix = np.tile([2**n for n in range(n_bits-1, -1, -1)], (n_samples, 1, 1))
    first_index = (prediction_1 * pot_matrix).sum(axis = 2).flatten()
    second_index = (prediction_2 * pot_matrix).sum(axis = 2).flatten()

    # Y hermitian * Codeword subspaces
    Y_C_1 = np.transpose(Y.conj(), axes = (0, 2, 1))@S_codebook[first_index, :, :]
    Y_C_2 = np.transpose(Y.conj(), axes = (0, 2, 1))@S_codebook[second_index, :, :]

    # Calculate Frobenius norm
    norm_1 = np.linalg.norm(Y_C_1, axis = (1,2), ord = 'fro')
    norm_2 = np.linalg.norm(Y_C_2, axis = (1,2), ord = 'fro')

    # Group calculated norms and codewords indexes predicted by NN
    total_norm = np.stack((norm_1, norm_2), axis = 1)
    total_index = np.stack((first_index, second_index), axis = 1)

    # Choose codeword index based on largest norm
    selected_index = total_index[np.arange(int(n_samples)), np.argmax(total_norm, axis = 1)]

    # Calculated SER with selected codewords
    SER = (selected_index != true_labels).sum()/n_samples
    return SER


def probabilities_post_processing(model, X, Y, true_labels, S_codebook, n_samples):
    """Post-processing based on first and second largest probabilities obtained
    by classification neural network.

    Args:
        model: Neural network.
        X: Preprocessed samples to make predictions.
        Y: Non-preprocessed samples.
        true_labels: Index of correct codeword of each sample.
        S_codebook: Codebook subspaces.
        n_samples: Number of samples.

    Returns:
        SER: Symbol Error Rate after post-processing.
    """

    # Codeword probabilities obtained by NN
    probabilities = model.predict(X)

    # Codewords with first and second largest probabilities
    first_index = np.argmax(probabilities, axis = 2).flatten()
    second_index = np.argsort(probabilities)[:, :, -2].flatten()

    # Y hermitian * Codeword subspaces
    Y_C_1 = np.transpose(Y.conj(), axes = (0, 2, 1))@S_codebook[first_index, :, :]
    Y_C_2 = np.transpose(Y.conj(), axes = (0, 2, 1))@S_codebook[second_index, :, :]

    # Calculate Frobenius norm
    norm_1 = np.linalg.norm(Y_C_1, axis = (1,2), ord = 'fro')
    norm_2 = np.linalg.norm(Y_C_2, axis = (1,2), ord = 'fro')

    # Group calculated norms and codewords indexes predicted by NN
    total_norm = np.stack((norm_1, norm_2), axis = 1)
    total_index = np.stack((first_index, second_index), axis = 1)

    # Choose codeword index based on largest norm
    selected_index = total_index[np.arange(int(n_samples)), np.argmax(total_norm, axis = 1)]

    # Calculated SER with selected codewords
    SER = (selected_index != true_labels).sum()/n_samples
    return SER