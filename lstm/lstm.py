import numpy as np
import datain

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)

def make_model(X, max_length):


    # instantiate the model
    embedding_vector_length = 32

    model = Sequential()
    model.add(Embedding(len(X), embedding_vector_length, input_length=max_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def fit_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, nb_epoch=3, batch_size=64)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    return scores

def extract_x_y(data):
    # slice out just one hot vectors and protein levels
    dict = data.loc[:, ['one_hots', 'p_levels']].to_dict('list')
    x = np.array(dict['one_hots'])
    y = np.array(dict['p_levels'])

    return x, y

def main():
    # load protein table + embedding matrices
    data = datain.main()

    X_train, Y_train = extract_x_y(data[0])
    X_test, Y_test = extract_x_y(data[1])

    # set the longest possible length to pad to (may want to automatically compute in future
    max_length = 400
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    model = make_model(X_train, max_length)

    print('Model summary '+str(model.summary()))

    scores = fit_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    print('Accuracy: %.2f%% ' % (scores[1*100]))



if __name__ == '__main__':
    main()

# blosum_62 = datain.load_data('box-data/BLOSUM62.csv')
# eigen = datain.load_data('box-data/protein_eigen.csv')
# hphob = datain.load_data('box-data/protein_hphob.csv')