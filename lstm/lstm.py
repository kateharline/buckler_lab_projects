import numpy
import datain

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility

def main():
    # load protein table + embedding matrices
    protein_seq = datain.main()
    blosum_62 = datain.load_data('box-data/BLOSUM62.csv')
    eigen = datain.load_data('box-data/protein_eigen.csv')
    hphob = datain.load_data('box-data/protein_hphob.csv')


if __name__ == '__main__':
    main()