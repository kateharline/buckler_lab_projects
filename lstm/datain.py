import numpy as np
import pandas as pd
import os
from sklearn import preprocessing as skp

# import control datasets for testing
import control as c




def load_data(filename):

    """read in the csv of gene sequences and RNAseq expression values
       returns the data as a pandas dataframe"""
    data = pd.read_csv(filepath_or_buffer=filename)
    # print(data.head(10))
    # print('data types '+str(data.dtypes))
    return data


def base_to_one_hot(seqs, encode_dict):
    """encode input data as one-hot vector
       adds one hot vector column to dataframe """
    newcol = []

    for seq in seqs:
        one_hot = []
        for letter in seq:
            one_hot.append(encode_dict[letter])
        newcol.append(one_hot)

    print('new col is '+str(newcol))

    return newcol


def main():

    os.chdir('/Users/kateharline/Desktop/buckler-lab')
    '''
    train/test synthetic data
    
    '''
    # how long is the sequence and how many are there... for synthetic data
    l = 400
    n = 10000

    synth = c.get_example('protein', n, l)
    heavy_As = c.get_example('heavy_As', n, l)

    encode_dict = load_data('box-data/protein_onehot.csv')

    '''
        for when I actually want to use real data

        # load the data from file
        x_data = load_data('X.csv')
        encode_dict = load_data('protein_onehot.csv')

    '''

    # convert the fasta file to one hot vectors
    synth_one_hots = base_to_one_hot(synth['protein_seqs'].tolist(), encode_dict.to_dict('list'))




if __name__ == '__main__':
    main()
