import numpy as np
import os
import pickle

# prevent warnings about CPU extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)

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
    model.add(LSTM(128, input_shape=(None, oh_length)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

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

def extract_x_y(data):
    '''
    reformat dataframe values into usable np arrays
    :param data: dataframe of sequence data and expression values
    :return: np arrays of x and y data
    '''
    # slice out just one hot vectors and protein levels
    dict = data.loc[:, ['one_hots', 'Protein_Leaf_Zone_3_Growth', 'Protein_Root_Meristem_Zone_5_Days']].to_dict('list')
    x = np.array(dict['one_hots'])
    y = np.array(dict['Protein_Leaf_Zone_3_Growth', 'Protein_Root_Meristem_Zone_5_Days'])

    return x, y

def main():
    # load protein table + embedding matrices
    train = pickle.load(open('train_encoded.pkl', 'rb'))
    test = pickle.load(open('test_encoded.pkl', 'rb'))

    X_train, Y_train = extract_x_y(train)
    X_test, Y_test = extract_x_y(test)

    # set the longest possible length to pad to (may want to automatically compute in future
    max_length = 400
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    model = make_model(X_train, max_length, 'protein')

    print('Model summary '+str(model.summary()))

    scores = fit_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    print('Accuracy: %.2f%% ' % (scores[1]*100))



if __name__ == '__main__':
    main()

# blosum_62 = datain.load_data('box-data/BLOSUM62.csv')
# eigen = datain.load_data('box-data/protein_eigen.csv')
# hphob = datain.load_data('box-data/protein_hphob.csv')