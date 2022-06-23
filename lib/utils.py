"""
    Functions to load and save data.
"""

from scripts.params import K, T, M, N, dataset_path, constellation, results_path, model_path, DL_model_name
import scipy.io as sio
import tensorflow as tf

def save_dataset_mat(dataset, filename):
    """Save dataset into MAT file.

    Args:
        dataset: Dictionary of the dataset.
        filename: Filename to save the dataset with.

    Returns:
        None
    """

    print("Saving dataset in", dataset_path, "...")
    sio.savemat(dataset_path + filename + ".mat", dataset)


def load_dataset(filename):
    """Load MAT file containing a dataset.

    Args:
        filename: Filename of the dataset.

    Returns:
        dataset: Dictionary of the dataset.
    """

    print("Loading dataset...")
    dataset = sio.loadmat(dataset_path + filename + ".mat")
    return dataset


def save_model_training(history, n_samples, train_SNR, prep, R, S, 
                        neurons, optimizer, lr, n_batch, filename, model_type = 'Categorical'):
    """Save neural network training information in a MAT file.

    Args:
        history: Evolution of metrics during neural network training.
        n_samples: Number of samples used for training the neural network.
        train_SNR: SNR of the training samples.
        prep: Preprocessing applied to training samples.
        R: Number of subspaces used for preprocessing.
        S: Subspaces used for preprocessing.
        neurons: Number of neurons in each neural network hidden layer.
        optimizer: TensorFlow optimizer for neural network.
        lr: Learning rate used during training.
        n_batch: Batch size for training.
        filename: Filename to save the training with.
        model_type: Categorical or soft bits neural network.

    Returns:
        None
    """

    dict = {'loss': history.history['loss'], 'val_loss': history.history['val_loss'],
            'n_samples': n_samples, 
            'constellation': constellation, 'prep': prep, 'R': R, 'S': S,
            'neurons': neurons, 'optimizer': optimizer, 'lr': lr, 
            'batch_size': n_batch, 'K': K, 'T': T, 'M': M, 'N': N}
    if train_SNR is not None:
        dict['train_SNR'] = train_SNR
    else:
         dict['train_SNR'] = 'None'
    if model_type == 'Categorical':
        dict['accuracy'] = history.history['accuracy']
        dict['val_accuracy'] = history.history['val_accuracy']
    elif model_type == 'Soft bits':
        dict['mse'] = history.history['mse']
        dict['val_mse'] = history.history['val_mse']
    print("Saving model training .mat in", results_path, "...")
    sio.savemat(results_path + filename + ".mat", dict)

def save_tensorflow_model(model, filename):
    """Save neural network.

    Args:
        model: Neural network.
        filename: Filename to save the model with.

    Returns:
        None
    """

    print("Saving DL model in", model_path, "...")
    model.save(model_path + filename)

def load_model(filename):
    """Load saved neural network.

    Args:
        filename: Filename of the neural network.

    Returns:
        DL_model: Neural network.
    """

    print("Loading DL model...")
    DL_model = tf.keras.models.load_model(model_path + filename)
    return DL_model


def save_SER_curve(SNR, SER_DL, SER_ML, SER_DL_post, num_sim, prep, R_prep, S_prep, train_SNR):
    """Save SNR-SER curve values in a MAT file.

    Args:
        SNR: SNR values.
        SER_DL: SER of neural network.
        SER_ML: SER of optimal detector.
        SER_DL_post: SER of neural network after posprocessing.
        num_sim: Array with the number of simulations for each SNR.
        prep: List of preprocessing of each neural network.
        R_prep: Number of subspaces for preprocessing of each neural network.
        S_prep: Preprocessing subspaces of each neural network.
        train_SNR: Training SNR of each neural network.

    Returns:
        None
    """

    filename = 'simulation_K=' + str(K) + ', T=' + str(T) + ', M=' + str(M) + ', N=' + str(N)
    dict = {'SNR': SNR, 'SER_DL': SER_DL, 'SER_DL_post': SER_DL_post,
            'num_sim': num_sim, 'prep': prep, 'R': R_prep, 'S': S_prep, 'train_SNR': train_SNR,
            'model_name': DL_model_name, 'constellation': constellation, 
            'K': K, 'T': T, 'M': M, 'N': N}
    if SER_ML is not None:
        dict['SER_ML'] = SER_ML
    print("Saving SER curve .mat in", results_path, "...")
    sio.savemat(results_path + filename + ".mat", dict)