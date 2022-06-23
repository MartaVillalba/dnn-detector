"""
    Functions to generate training data.
"""

from scripts.params import K, T, M, N, constellation, codebook_path
import numpy as np
import scipy.io as sio

def dataset_generation(n_samples, SNR, clip_rate = None):
    """Dataset generation based on Y = X*H + N (without clipping) 
    or Y = c(X)*H + N (with clipping).

    Args:
        n_samples: Number of samples in the dataset.
        SNR: Signal-to-noise ratio of the samples.
        clip_rate: Clipping fraction, if desired.

    Returns:
        dataset: Dictionary containing generated samples.
    """

    print("Generating dataset...")
    # Load constellation
    filename = "codebook_K" + str(K) + "_T" + str(T) + "_M" + str(M) + "_" + constellation
    dataset_codebook = sio.loadmat(codebook_path + filename + ".mat")
    Codebook_aux = dataset_codebook['C']
    Codebook = np.zeros( (K, T, M), dtype = complex)
    for i in range(K):
        Codebook[i,:,:] = Codebook_aux[:,:,i]

    if clip_rate:
        Codebook = clipping(Codebook, clip_rate)

    # Noise generation
    if SNR:
        NoiseVar = (M/T) * 10**(-SNR/10)
        Noise = np.sqrt(NoiseVar/2) * np.random.randn(n_samples, T, N) + 1j * np.sqrt(NoiseVar/2) * np.random.randn(n_samples, T, N)

    # Dataset generation
    y = np.random.randint(K, size = n_samples) # Tx codewords uniformly distributed
    S = Codebook[y, :, :] # classes --> [1,K], array indeces --> [0,K-1]

    H = np.random.randn(n_samples, M, N)/np.sqrt(2) + 1j*np.random.randn(n_samples, M, N)/np.sqrt(2) # Rayleigh MIMO channel (fixed during coherence block)

    if SNR:
        X = S @ H + Noise # Rx codewords (w/ noise)
    else:
        X = S @ H

    dataset = {'X': X, 'y': y, 'T': T, 'M': M, 'N': N, 'K': K}
    if SNR:
        dataset['SNR'] = SNR
    else:
        dataset['SNR'] = float('inf')

    return dataset


def dataset_generation_variable_noise(n_samples_array, SNR_array):
    """Dataset generation based on Y = X*H + N with several SNR samples.

    Args:
        n_samples_array: List containing the number of samples for each SNR.
        SNR_array: List containing SNR values.

    Returns:
        dataset: Dictionary containing generated samples.
    """

    print("Generating dataset with variable SNR...")
    # Load constellation
    filename = "codebook_K" + str(K) + "_T" + str(T) + "_M" + str(M) + "_" + constellation
    dataset_codebook = sio.loadmat(codebook_path + filename + ".mat")
    Codebook_aux = dataset_codebook['C']
    Codebook = np.zeros( (K, T, M), dtype = complex)
    for i in range(K):
        Codebook[i,:,:] = Codebook_aux[:,:,i]

    # Number of total training samples
    dataset_size = n_samples_array.sum() 

    # Dataset generation
    NoiseVar = (M/T) * 10**(-SNR_array/10)

    X_total = np.zeros((dataset_size, T, N)) + 1j * np.zeros((dataset_size, T, N))
    y_total = np.zeros((dataset_size), dtype = np.int32)
    SNR_total = np.zeros((dataset_size))

    for i in range(SNR_array.size):

        Noise = np.sqrt(NoiseVar[i]/2) * np.random.randn(n_samples_array[i], T, N) 
        + 1j * np.sqrt(NoiseVar[i]/2) * np.random.randn(n_samples_array[i], T, N)

        y = np.random.randint(K, size = n_samples_array[i]) # Tx codewords uniformly distributed

        S = Codebook[y,:,:] # classes --> [1,NumCodewords], array indeces --> [0,NumCodewords-1]

        H = np.random.randn(n_samples_array[i], M, N)/np.sqrt(2) 
        + 1j*np.random.randn(n_samples_array[i], M, N)/np.sqrt(2) # Rayleigh MIMO channel (fixed during coherence block)

        X = np.zeros((n_samples_array[i], T, N)) + 1j * np.zeros((n_samples_array[i], T, N))
        for k in range(n_samples_array[i]):
            X[k,:,:] = S[k, :, :] @ H[k,:,:] + Noise[k, :, :] # Rx codewords
        
        start = n_samples_array[0:i].sum()
        end = start + n_samples_array[i]
        y_total[start:end] = y
        X_total[start:end, :, :] = X
        SNR_total[start:end] = SNR_array[i]

    ind = np.arange(dataset_size)
    np.random.shuffle(ind)
    y_total = y_total[ind]
    X_total = X_total[ind, :, :]
    SNR_total = SNR_total[ind]
    dataset = {'X': X_total, 'y': y_total, 'T': T, 'M': M, 'N': N, 'K': K, 'SNR': SNR_array, 'SNR_total': SNR_total}
    return dataset


def clipping(X, clip_rate):
    """Apply clipping to an individual signal sample.

    Args:
        X: Signal to clip.
        clip_rate: Clipping fraction.

    Returns:
        X_clip: Clipped sample.
    """

    X_abs = np.abs(X)
    X_max = X_abs.max()
    saturation = clip_rate*X_max
    sat_values = X_abs >= saturation
    nonsat_values = X_abs < saturation
    X_clip = X * nonsat_values + saturation * sat_values * X/X_abs
    return X_clip