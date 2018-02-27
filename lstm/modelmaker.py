import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as cor
from keras.models import Model
from keras.layers import Activation, MaxPooling1D, Flatten, BatchNormalization, Input, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.optimizers as optimizers
import keras.backend as backend
import matplotlib.pyplot as plt
import datain as d
import h5py
import platform

# prevent warnings about CPU extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D
# from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)


def prediction_accuracy(y_true, y_pred):
    '''calculate the prediction accuracy of the model
    :param y_true: float known y expression value
    :param y_pred:float y value predicted by the model
    :return:float accuracy score
    '''
    c12 = backend.sum((y_true - backend.mean(y_true)) * (y_pred - backend.mean(y_pred)))
    c11 = backend.sum(backend.square(y_true - backend.mean(y_true)))
    c22 = backend.sum(backend.square(y_pred - backend.mean(y_pred)))
    return c12 / backend.sqrt(c11 * c22)

def y_pred_mean(y_true, y_pred):
    return backend.mean(y_pred)


def fit_and_evaluate(model, model_dir, model_name, y_train, y_val, y_test, protein_train, protein_test, protein_val,
                     make_checkpoints=True):
    min_delta = 0
    patience = 5
    batch_size = 256
    n_epochs = 100
    # Training
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')]

    if make_checkpoints:
        callbacks.append(
            ModelCheckpoint(filepath=os.path.join(model_dir, model_name + '__epoch={epoch:02d}.h5'), period=10))

    fit = model.fit([protein_train], [y_train],
                    validation_data=([protein_val], [y_val]),
                    batch_size=batch_size, epochs=n_epochs, callbacks=callbacks)

    # Saving fitted model
    try:
        model.save(os.path.join(model_dir, model_name) + '.h5')
    except ValueError:
        warnings.warn('Model could not be saved')

    # Validation
    ypred_train = model.predict([protein_train])
    ypred_val = model.predict([protein_val])
    ypred_test = model.predict([protein_test])

    y_train_flat = ypred_train.flatten()
    y_val_flat = ypred_val.flatten()
    y_test_flat = ypred_test.flatten()

    cor_train = cor(y_train, y_train_flat)
    cor_val = cor(y_val, y_val_flat)
    cor_test = cor(y_test, y_test_flat)

    return fit, [y_train, ypred_train, cor_train], [y_test, ypred_test, cor_test], [y_val, ypred_val, cor_val]


def plot_stats(fit, model_name, model_dir, y_train, y_val, selected_tissue):
    model_metric = prediction_accuracy
    metric_name = model_metric.__name__
    plt_metric_name = metric_name.replace('_', ' ').capitalize()
    # Plot training history
    accuracy_train = fit.history[list(fit.history)[-1]]
    accuracy_val = fit.history[list(fit.history)[1]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(accuracy_train, color='g', label='Training')
    ax.plot(accuracy_val, color='b', label='Validation')
    ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
           xlabel='Epoch',
           ylabel=plt_metric_name)
    ax.legend(loc='best')
    fig.savefig(os.path.join(model_dir, model_name + '_history--' + selected_tissue + '.png'))

    # Training and validation correlation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_train[0], y_train[1], color='g',
               label='Training: r=' + str(np.round(y_train[2], 3)), alpha=0.15)
    ax.scatter(y_val[0], y_val[1], color='b',
               label='Validation: r=' + str(np.round(y_val[2], 3)), alpha=0.15)
    ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
           xlabel='Observed',
           ylabel='Predicted')
    ax.legend(loc='best')
    fig.savefig(os.path.join(model_dir, model_name + '-predicting_validation_set--' + selected_tissue + '.png'))

    return accuracy_train, accuracy_val

def main(model_name, model_dir, t_params, X_train, Y_train, X_test, Y_test, X_val, Y_val, tissue='Protein_Leaf_Zone_3_Growth'):
    # t_params : learning_rate, num_layers, units_per_layer

    max_length = len(X_train[0])

    model = make_model(X_train, max_length, 'protein', t_params[1], t_params[2])
    fit, y_train, y_test, y_val = fit_and_evaluate(model, model_dir, model_name, Y_train, Y_val, Y_test, X_train,
                                                   X_test,
                                                   X_val)
    accuracy_train, acc_test = plot_stats(fit, model_name, model_dir, y_train, y_val, tissue)

    print('Model summary ' + str(model.summary()))



if __name__ == '__main__':
    main()