import numpy as np
import datain
import os

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


def make_model(X, max_length, seq_type):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein':21, 'na':4}
    oh_length = oh_lengths[seq_type]

    print('length '+str(oh_length))

    # instantiate the model
    embedding_vector_length = 32

    model = Sequential()
    # model.add(Embedding(21, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(32, 3, activation='relu', input_shape=(max_length, oh_length)))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model


def fit_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    '''
    :param model: keras model
    :param X_train: np array
    :param Y_train: np array
    :param X_test: np array
    :param Y_test: np array
    :return: score data from keras
    '''
    model.fit(X_train, Y_train, epochs=10, batch_size=16)
    scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=16)
    return scores


def main():
    # load protein table + embedding matrices
    data = datain.main()

    X_train, Y_train = extract_x_y(data[0])
    X_test, Y_test = extract_x_y(data[1])

    # set the longest possible length to pad to (may want to automatically compute in future
    max_length = 399
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    model = make_model(X_train, max_length, 'protein')

    print('Model summary '+str(model.summary()))

    scores = fit_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    print('Accuracy: %.2f%% ' % (scores[1]*100))



if __name__ == '__main__':
    main()