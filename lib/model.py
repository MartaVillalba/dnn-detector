"""
    Functions to define, train and evaluate a neural network model in TensorFlow.
"""

from scripts.params import K, T, M, N
import tensorflow as tf
from keras import layers
from keras import models
import numpy as np
from sklearn.model_selection import train_test_split

def model_definition(n_neurons, input_shape =  (1, 2*T*M), dropout = None, 
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001), 
                    reg_dict = {}, init_dict = {}):
    """Neural network definition based on multiclass classification.

    Args:
        n_neurons: Number of neurons in each hidden layer.
        input_shape: Input dimensions for input layer.
        dropout: Dropout fraction in hidden layers.
        optimizer: TensorFlow optimizer for neural network.
        reg_dict: Regularization configuration.
        init_dict: Weight initializer configuration.

    Returns:
        model: Defined and compiled neural network.
    """

    print("Defining DL model...")
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape = input_shape))
    for n in n_neurons:
        model.add(layers.Dense(n, activation='relu', **reg_dict, **init_dict))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(K, activation='softmax', **init_dict))
    model.summary()

    model.compile(loss = tf.keras.losses.categorical_crossentropy, 
                optimizer = optimizer, 
                metrics = ['accuracy'])

    return model


def model_definition_soft_bits(n_neurons, input_shape, dropout = None, 
                               optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001), 
                               reg_dict = {}, init_dict = {}):
    """Neural network definition based on regression (soft bits).

    Args:
        n_neurons: Number of neurons in each hidden layer.
        input_shape: Input dimensions for input layer.
        dropout: Dropout fraction in hidden layers.
        optimizer: TensorFlow optimizer for neural network.
        reg_dict: Regularization configuration.
        init_dict: Weight initializer configuration.

    Returns:
        model: Defined and compiled neural network.
    """

    print("Defining DL model...")
    n_bits = int(np.log2(K))
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape = input_shape))
    for n in n_neurons:
        model.add(layers.Dense(n, activation='relu', **reg_dict, **init_dict))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_bits, activation='sigmoid', **init_dict))
    model.summary()

    model.compile(loss = tf.keras.losses.MeanSquaredError(), 
                optimizer = optimizer, 
                metrics = ['mse'])

    return model


def model_training(model, X, y, val_split, n_epochs, n_batch):
    """Neural network training.

    Args:
        model: Neural network to train.
        X: Input samples.
        y: Output samples.
        val_split: Fraction of data destinated to test the neural network during training.
        n_epochs: Duration of training in epochs.
        n_batch: Batch size for training.

    Returns:
        model: Trained neural network.
        history: Evolution of metrics during training.
    """

    print("Training DL model...")
    # Train-test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val_split, random_state = 0)

    # Model training
    history = model.fit(X_train, y_train, batch_size = n_batch, epochs = n_epochs, validation_data = (X_test, y_test))

    return model, history


def evaluate_model(model, X, y):
    """Multiclass classification neural network evaluation in terms of SER.

    Args:
        model: Neural network to evaluate.
        X: Input samples.
        y: Output samples.

    Returns:
        SER: Symbol Error Rate.
    """

    results = model.evaluate(x = X, y = y, verbose = 0)
    SER = 1 - results[1]
    return SER


def evaluate_model_soft_bits(model, X, true_labels, n_samples):
    """Regression (soft bits) neural network evaluation in terms of SER.

    Args:
        model: Neural network to evaluate.
        X: Input samples.
        true_labels: Output samples.
        n_samples: Number of samples to evaluate the neural network on.

    Returns:
        SER: Symbol Error Rate.
        prediction: Output of neural network after evaluation.
        prediction_bits: Output of neural network converted to bits (0 or 1).
    """

    prediction = model.predict(X)
    prediction_bits = np.array((prediction-0.5) > 0, dtype=np.int8)
    correct_predictions = [int(sample.all()) for sample in (prediction_bits == true_labels)]
    SER = 1 - (np.sum(correct_predictions)/n_samples)
    return SER, prediction, prediction_bits