import numpy as np
import pandas as pd


def load_data(filename):

    """read in the csv of gene sequences and RNAseq expression values"""
    data = pd.read_csv(filepath_or_buffer=filename)
    print(data.head(10))

    return data


def base_to_one_hot(df):
    """encode input data as one-hot vector"""


    return None


def main():
    # load the data from file
    data = load_data('/Users/kateharline'
                                          '/Desktop/buckler_lab/kak268/'
                                          'preprocessed_exp_w_sequence/'
                                          'AllTissues_exp_counts_and_seq_from_Zea_mays.AGPv4.34.1kb_upstr_of_1st_5pUTR.first_tscript_per_gene.csv')
    # convert the fasta file to one hot vectors
    base_one_hots = base_to_one_hot(data)


    pass

if __name__ == '__main__':
    main()
