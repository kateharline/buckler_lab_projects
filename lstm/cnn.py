import platform
import os
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as cor
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import keras.layers.advanced_activations as advanced_activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.initializers import he_normal
import keras.optimizers as optimizers
import keras.backend as backend
import keras.layers
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# prevent warnings about CPU extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D
# from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)

def extract_x_y(data):
    '''
    reformat dataframe values into usable np arrays
    :param data: dataframe of sequence data and expression values
    :return: np arrays of x and y data
    '''
    # slice out just one hot vectors and protein levels
    dict = data.loc[:, ['one_hots', 'p_levels']].to_dict('list')
    x = np.array(dict['one_hots'])
    y = np.array(dict['p_levels'])

    return x, y

def prediction_accuracy(y_true, y_pred):
    '''calculate the prediction accuracy of the model
    :param y_true: float known y expression value
    :param y_pred:float y value predicted by the model
    :return:float accuracy score
    '''
    c12 = backend.sum((y_true - backend.mean(y_true)) * (y_pred - backend.mean(y_pred)))
    c11 = backend.sum(backend.square(y_true - backend.mean(y_true)))
    c22 = backend.sum(backend.square(y_pred - backend.mean(y_pred)))
    return c12/backend.sqrt(c11*c22)


# Protein sequence scan
def protein_scan(input_sequence, cnn_layers=4, fcn_layers=1):
    '''
    use the functional API to instantiate layers in CNN
    :param input_sequence: ???
    :param cnn_layers: int number of convolutional layers
    :param fcn_layers: int number of fully connected layers
    :return: model with layers applied
    '''
    x = Conv1D(64, kernel_size=5, padding='valid')(input_sequence)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(padding='same')(x)
    x = Dropout(0.25)(x)

    for _ in range(1, cnn_layers):
        x = Conv1D(64, kernel_size=5, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)
        x = Dropout(0.25)(x)

    x = Flatten()(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)

    return x

def fc_apply(motifs):
    #   FC layers on concatenated representations
    expression = Dense(64, activation='relu')(motifs)
    expression = Dense(64, activation='relu')(expression)

    #   Output
    expression = Dense(1, activation='relu')(expression)
    return expression


def make_model(protein, max_length, seq_type):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein':21, 'na':5}
    oh_length = oh_lengths[seq_type]
    metrics = prediction_accuracy()

    print('length '+str(oh_length))

    # instantiate the model
    conv_protein = protein_scan(protein)
    fcs = fc_apply(conv_protein)

    model = Model(inputs=[protein],
              outputs=[fcs],
              name='protein_level')

    # Inspection
    model.summary()
    print('Output shape: ' + str(model.output_shape))

    # Compilation
    model.compile(optimizer=optimizers.Adam(),
                  loss='mse',
                  metrics=[metrics])

    return model


def fit_and_evaluate(model, model_dir, model_name, y_train, y_val, y_test, protein_train, protein_test, protein_val,
                     make_checkpoints=True):
    min_delta = 0
    patience = 5
    batch_size = 256
    n_epochs = 100
    # Training
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')]

    if make_checkpoints:
        callbacks += ModelCheckpoint(filepath=os.path.join(model_dir, model_name + '__epoch={epoch:02d}.h5'), period=10)

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

    cor_train = cor(y_train, ypred_train)[0][0]
    cor_val = cor(y_val, ypred_val)[0][0]
    cor_test = cor(y_test, ypred_test)[0][0]

    return [y_train, ypred_train, cor_train], [y_test, ypred_test, cor_test], [y_val, ypred_val, cor_val]

def plot_stats(fit, model_name, model_dir, y_train, y_val, plt_metric_name, selected_tissue):
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


def main():
    train = pd.read_csv('train_encoded.csv')
    test = pd.read_csv('test_encoded.csv')
    val = pd.read_csv('val_encoded.csv')

    X_train, Y_train = extract_x_y(train)
    X_val, Y_val = extract_x_y(val)
    X_test, Y_test = extract_x_y(test)

    max_length = len(X_train['one_hots'][0])

    model = make_model(X_train[['one_hots']], max_length, 'protein')

    print('Model summary '+str(model.summary()))

    scores = fit_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    print('Accuracy: %.2f%% ' % (scores[1]*100))



if __name__ == '__main__':
    main()