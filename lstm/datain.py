import numpy as np
import pandas as pd
import os
from sklearn import preprocessing as skp




def load_data(filename):

    """read in the csv of gene sequences and RNAseq expression values
       returns the data as a pandas dataframe"""
    data = pd.read_csv(filepath_or_buffer=filename)
    # print(data.head(10))
    # print('data types '+str(data.dtypes))
    return data


def get_one_hots(df, cis_size, seq_label, encode_dict):
    """ helper function that converts each sequence entry to a one hot vector
    returns array of one-hots the same height as the df"""
    # dimensions of dataframe
    rows = df.shape[0]

    return one_hots

def base_to_one_hot(df, cis_size, encode_dict):
    """encode input data as one-hot vector
       adds one hot vector column to dataframe """
    newcol = get_one_hots(df, cis_size, seq_label='fasta', encode_dict)
    print('new col is '+str(newcol))
    df.assign(seq_one_hot=newcol)

    return newcol


def main():
    # what promoter size did you pick
    us = 1001
    ds = 500
    size = us + ds

    os.chdir('/Users/kateharline/Desktop/buckler-lab')

    # load the data from file
    x_data = load_data('X.csv')
    encode_dict = load_data('protein_onehot.csv')


    # convert the fasta file to one hot vectors
    base_one_hots = base_to_one_hot(x_data, size, encode_dict)


    print('how many cols '+str(base_one_hots.shape[1]))

    print(base_one_hots.head(10))


    return None

if __name__ == '__main__':
    main()
