"""
    Monte Carlo simulations.
"""

from scripts.params import K, T, M, N, constellation, codebook_path, ML_path, max_sim
import numpy as np
import scipy.io as sio
import time
from lib.dataset import clipping
from lib.processing import preprocessing, generate_subspaces, probabilities_post_processing, soft_bits_post_processing
from lib.model import evaluate_model, evaluate_model_soft_bits
from lib.plot import plot_SER_curve
from lib.utils import save_SER_curve


def mc_simulation(SNR, num_sim, DL_model_array, prep, label_prep, R_prep, S_prep, train_SNR,
                    clip_rate = None, load_ML = True, post_processing = True, variable_SNR = False):
    """Monte Carlo simulation to evaluate neural network SNR-SER curve.

    Args:
        SNR: Array of SNR values.
        num_sim: Array with the number of simulations for each SNR.
        DL_model_array: List of neural networks to evaluate (up to 6).
        prep: List of preprocessing of each neural network.
        label_prep: List of label preprocessing of each neural network.
        R_prep: Number of subspaces for preprocessing of each neural network.
        S_prep: Preprocessing subspaces of each neural network.
        train_SNR: Training SNR of each neural network.
        clip_rate: Clipping fraction, if desired.
        load_ML: Include optimal detector in SNR-SER curve or not.
        post_processing: Apply posprocessing to neural networks predictions or not.
        variable_SNR: If the neural networks have been trained with variable SNR or not.

    Returns:
        None
    """
    # Load constellation
    filename = "codebook_K" + str(K) + "_T" + str(T) + "_M" + str(M) + "_" + constellation
    constel = sio.loadmat(codebook_path + filename + ".mat")
    Codebook_aux = constel['C']
    Codebook = np.zeros( (K, T, M), dtype = complex)
    for i in range(K):
        Codebook[i,:,:] = Codebook_aux[:,:,i]

    if clip_rate:
        Codebook = clipping(Codebook, clip_rate)

    if load_ML:
        if clip_rate:
            filename = 'ML_clipping_' + str(clip_rate) + '_K=' + str(K) + '_T=' + str(T) + '_M=' + str(M) + '_N=' + str(N) + '_' + constellation
        else:
            filename = 'ML_K=' + str(K) + '_T=' + str(T) + '_M=' + str(M) + '_N=' + str(N) + '_' + constellation
        ML = sio.loadmat(ML_path + filename + ".mat")
        SER_ML = ML['SER_ML'].flatten()
    else:
        SER_ML = None

    # Neural network variables
    SER_DL = []
    DL_time = []
    DL_pre_time = []
    if post_processing:
        SER_DL_post = []
        DL_post_time = []
        # Codebook subspaces for post-processing
        S_codebook = generate_subspaces('Codebook', K)
    else:
        SER_DL_post = None
    for i in range(len(DL_model_array)):
        SER_DL.append(np.zeros((SNR.size)))
        DL_time.append(0)
        DL_pre_time.append(0)
        if post_processing:
            SER_DL_post.append(np.zeros((SNR.size)))
            DL_post_time.append(0)

    # Simulation
    NoiseVar = (M/T) * 10 ** (-SNR / 10) # noise variance

    print("Simulating...\n")
    for cc in range(SNR.size):
        print("SNR = " + str(SNR[cc]) + " dB")
        print("------------------------------")
        VarNoise = NoiseVar[cc]

        n_samples = int(num_sim[cc])
        if n_samples > max_sim:
            # Número de bloques
            n_bloq = int(n_samples/max_sim)
            # Tamaño de cada bloque
            bloq_sim = int(n_samples/n_bloq)
        else:
            # Sin bloques
            n_bloq = 1
            bloq_sim = n_samples

        for bloq in range(n_bloq):
            print('Block ' + str(bloq+1) + ' of ' + str(n_bloq) + '...')
            labels = np.random.randint(K, size = bloq_sim)
            X = Codebook[labels, :, :]
            H = (np.random.randn(bloq_sim, M, N) / np.sqrt(2) + 1j * np.random.randn(bloq_sim, M, N) / np.sqrt(2))
            Noise = (np.sqrt(VarNoise / 2) * np.random.randn(bloq_sim, T, N) + 1j * np.sqrt(VarNoise / 2) * np.random.randn(bloq_sim, T, N))
        
            Y = X @ H + Noise

            if variable_SNR:
                SNR_total = np.repeat(SNR[cc], bloq_sim)

            # Deep learning detectors
            for i in range(len(DL_model_array)):
                # Preprocessing samples
                start_pre = time.time()
                if variable_SNR:
                    X, y = preprocessing(Y, labels, bloq_sim, prep[i], R_prep[i], S_prep[i], label_prep[i], SNR_total)
                else:
                    X, y = preprocessing(Y, labels, bloq_sim, prep[i], R_prep[i], S_prep[i], label_prep[i])
                end_pre = time.time()
                DL_pre_time[i] += end_pre - start_pre

                # Evaluate model
                if label_prep[i] == 'Categorical':
                    # Without post-processing
                    start_ev = time.time()
                    SER = evaluate_model(DL_model_array[i], X, y)
                    end_ev = time.time()
                    # With post-processing
                    if post_processing:
                        start_post = time.time()
                        SER_post = probabilities_post_processing(DL_model_array[i], X, Y, labels, S_codebook, bloq_sim)
                        end_post = time.time()
                elif label_prep[i] == 'Soft bits':
                    # Without post-processing
                    start_ev = time.time()
                    SER, prediction, prediction_bits = evaluate_model_soft_bits(DL_model_array[i], X, y, bloq_sim)
                    end_ev = time.time()
                    # With post-processing
                    if post_processing:
                        start_post = time.time()
                        SER_post = soft_bits_post_processing(prediction, prediction_bits, Y, labels, S_codebook, bloq_sim)
                        end_post = time.time()
                # Save results
                DL_time[i] += end_ev - start_ev
                SER_DL[i][cc] += SER/n_bloq
                DL_post_time[i] += end_post - start_post
                SER_DL_post[i][cc] += SER_post/n_bloq

        for i in range(len(DL_model_array)):
            print("\tSER DL " + str(i + 1) + " = " + str(SER_DL[i][cc]) + "\n")
            if post_processing:
                print("\tSER DL " + str(i + 1) + " post-processing = " + str(SER_DL_post[i][cc]) + "\n")

    print("------------------------------")

    for i in range(len(DL_model_array)):
        print("DL " + str(i + 1) + " evaluate time required = ", str(DL_time[i]), "seconds \n")
        print("DL " + str(i + 1) + " preprocessing time required = ", str(DL_pre_time[i]), "seconds \n")
        if post_processing:
            print("DL " + str(i + 1) + " post-processing time required = ", str(DL_post_time[i]), "seconds \n")

    plot_SER_curve(SNR, SER_DL, SER_ML, SER_DL_post)
    save_SER_curve(SNR, SER_DL, SER_ML, SER_DL_post, num_sim, prep, R_prep, S_prep, train_SNR)


def mc_simulation_optimal_detector(SNR, num_sim, clip_rate = None):
    """Monte Carlo simulation to obtain optimal detector SNR-SER curve.

    Args:
        SNR: Array of SNR values.
        num_sim: Array with the number of simulations for each SNR.
        clip_rate: Clipping fraction, if desired.

    Returns:
        SER_ML: Symbol Error Rate array.
    """

    # Load constellation
    codebook_path = "data/Codebooks/"
    filename = "codebook_K" + str(K) + "_T" + str(T) + "_M" + str(M) + "_" + constellation
    constel = sio.loadmat(codebook_path + filename + ".mat")
    Codebook = constel['C']

    if clip_rate:
        saturation = clip_rate*np.abs(Codebook).max()

    # Optimal detector variables
    SER_ML = np.zeros((SNR.size))
    PCodebook = np.zeros((K, T, T)) + 1j * np.zeros((K, T, T))
    for i in range(K):
        codeword_i = Codebook[:,:,i]
        PCodebook[i,:,:] = codeword_i @ codeword_i.conj().T # projection matrices

    # Simulation
    NoiseVar = (M/T) * 10 ** (-SNR / 10) # noise variance

    print("Simulating...\n")
    for cc in range(SNR.size):
        print("SNR = " + str(SNR[cc]) + " dB")
        print("------------------------------")

        VarNoise = NoiseVar[cc]

        for dd in range(int(num_sim[cc])):
            
            # Generate sample
            TX_Codeword = np.random.randint(K)
            X = Codebook[:, :, TX_Codeword] # TX Codeword
            if clip_rate:
                X_abs = np.abs(X)
                sat_values = X_abs >= saturation
                nonsat_values = X_abs < saturation
                X = X * nonsat_values + saturation * sat_values * X/X_abs
            H = (np.random.randn(M, N) / np.sqrt(2)
            + 1j * np.random.randn(M, N) / np.sqrt(2)) # Rayleigh MIMO channel (fixed during the coherence block)
            Noise = (np.sqrt(VarNoise / 2) * np.random.randn(T, N) 
                    + 1j * np.sqrt(VarNoise / 2) * np.random.randn(T, N)) # AWGN Noise

            Y = X @ H + Noise
                
            # Grassmannian ML detector (Optimal)
            dist = (Y.conj().T @ PCodebook @ Y).trace(axis1 = 1, axis2 = 2).real
            subspace = np.argmax(dist) # decision
            SER_ML[cc] = SER_ML[cc] + (subspace != TX_Codeword) # count errors

        print("\tSER ML = " + str(SER_ML[cc] / num_sim[cc]) + "\n")

    SER_ML = SER_ML / num_sim
    return SER_ML
    