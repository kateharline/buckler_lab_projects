import numpy as np
import pandas as pd


def load_data(filename):

    """read in the csv of gene sequences and RNAseq expression values
       returns the data as a pandas dataframe"""
    data = pd.read_csv(filepath_or_buffer=filename)
    # print(data.head(10))
    # print('data types '+str(data.dtypes))
    return data


def get_one_hots(df, cis_size, seq_label):
    """ helper function that converts each sequence entry to a one hot vector
    returns array of one-hots the same height as the df"""
    # dimensions of dataframe
    rows = df.shape[0]
    # initialize
    one_hots = np.zeros((rows, cis_size, 5))

    # fill array with one hot versions of each sequence
    for i in range(rows):
        fasta = df[seq_label][i]

    return None

def base_to_one_hot(df, cis_size):
    """encode input data as one-hot vector
       adds one hot vector column to dataframe """
    newcol = get_one_hots(df, cis_size, seq_label='fasta')
    df.assign(seq_one_hot=newcol)

    return df


def main():
    # what promoter size did you pick
    us = 1001
    ds = 0
    cis_size = us + ds
    # load the data from file
    data = load_data('/Users/kateharline'
                                          '/Desktop/buckler-lab/kak268/'
                                          'preprocessed_exp_w_sequence/'
                                          'AllTissues_exp_counts_and_seq_from_Zea_mays.AGPv4.34.1kb_upstr_of_1st_5pUTR.first_tscript_per_gene.csv')
    # convert the fasta file to one hot vectors

    base_one_hots = base_to_one_hot(data, cis_size)


    pass

if __name__ == '__main__':
    main()
