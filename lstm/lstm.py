import numpy
import datain

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility

def main():
    # run data in to synthesize or load data
    protein_seq = datain.main()



if __name__ == '__main__':
    main()