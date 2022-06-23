"""
    Functions to plot results.
"""

from scripts.params import K, T, M, N, plot_path, DL_model_name
import matplotlib.pyplot as plt

def plot_dataset(dataset, type, R, filename):
    """Plot dataset samples.

    Args:
        dataset: Dictionary containing signal samples.
        type: Preprocessing applied to samples.
        R: Number of subspaces used for preprocessing.
        filename: Filename to save the plot with.

    Returns:
        None
    """

    X = dataset['X']
    if type == 'Principal angles':
        plt.figure()
        plt.matshow(X[0:M*R, :, :].reshape(M*R, M*R))
        plt.colorbar()
        plt.title('Training dataset')
        plt.ylabel('Sample')
        plt.xlabel('Principal angles')
        plt.savefig(plot_path + filename + '.png')
        plt.clf()


def plot_model_training(history, filename, model_type = 'Categorical'):
    """Plot neural network training.

    Args:
        history: Training history.
        filename: Filename to save the plot with.
        model_type: Categorical or soft bits model.

    Returns:
        None
    """

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('K=' + str(K) + ', T=' + str(T) + ', M=' + str(M) + ', N=' + str(N))
    subfig = fig.add_subplot(211)
    if model_type == 'Categorical':
        subfig.plot(history.history['accuracy'], label="Train")
        subfig.plot(history.history['val_accuracy'], label="Validation")
        subfig.set_ylabel('Accuracy')
    elif model_type == 'Soft bits':
        subfig.plot(history.history['mse'], label="Train")
        subfig.plot(history.history['val_mse'], label="Validation")
        subfig.set_ylabel('MSE')
    subfig.set_xlabel('Epoch')
    subfig.set_yscale('log')
    subfig.legend(loc='lower right')
    subfig = fig.add_subplot(212)
    subfig.plot(history.history['loss'], label="Train")
    subfig.plot(history.history['val_loss'], label="Validation")
    subfig.set_ylabel('Loss')
    subfig.set_xlabel('Epoch')
    subfig.set_yscale('log')
    subfig.legend(loc='upper right')
    fig.savefig(plot_path + filename + '.png')
    fig.clf()


def plot_SER_curve(SNR, SER_DL, SER_ML, SER_DL_post):
    """Plot SNR-SER curve for neural network and optimal detector.

    Args:
        SNR: SNR values.
        SER_DL: SER of neural network.
        SER_ML: SER of optimal detector.
        SER_DL_post: SER of neural network after posprocessing.

    Returns:
        None
    """

    filename = 'simulation_K=' + str(K) + ', T=' + str(T) + ', M=' + str(M) + ', N=' + str(N)
    plt.figure()
    if SER_ML is not None:
        plt.semilogy(SNR, SER_ML, 'ro-', ms=5, lw=1.5, label='ML detector')
        colors = ['bs-', 'gs-', 'ks-', 'cs-', 'ms-', 'ys-']
    else:
        colors = ['rs-', 'bs-', 'gs-', 'ks-', 'cs-', 'ms-', 'ys-']
    for i in range(len(SER_DL)):
        plt.semilogy(SNR, SER_DL[i], colors[i], ms=5, lw=1.5, label=DL_model_name[i])
        if SER_DL_post is not None:
            plt.semilogy(SNR, SER_DL_post[i], colors[i] + '-', ms=5, lw=1.5, label='Post-processed ' + DL_model_name[i])
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    plt.title("T = " + str(T) + ", M = " + str(M) + ", N = " + str(N) + ", K = " + str(K))
    plt.legend(prop={'size': 7})
    plt.grid(axis = 'both', which = 'both', ls = '--', lw = 0.5)
    print("Saving SER-SNR surve in", plot_path, "...")
    plt.savefig(plot_path + filename + '.png')
    plt.clf()