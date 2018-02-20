# external libraries
import numpy as np
import pandas as pd
import os
import pickle

# import control datasets for testing
import control as c

####--------------making matrices-------------#############
def txt_to_csv():
    '''
    convert txt file to csv to load as a dataframe etc
    :return: NA, outputs file as csv
    '''

    os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')


    with open('BLOSUM62.txt') as f:
        with open('blosum62.csv', 'w') as out:

            for line in f:
                new_line = ','.join(line.split())
                out.write(new_line)
                out.write('\n')


##### make hydrophobicity matrix -- how to denote *


# column names for data frame
def make_hphob_matrix():
    '''
    compute differences in hydrophobicity between amino acids
    :return: NA
    '''
    hydro_values = {'I': 4.92, 'L': 4.92, 'V': 4.04, 'P': 2.98, 'F': 2.98, 'M': 2.35,
                    'W': 2.33, 'A': 1.81, 'C': 1.28, 'G': 0.94, 'Y': -0.14, 'T': -2.57,
                    'S': -3.40, 'H': -4.66, 'Q': -5.54, 'K': -5.55, 'N': -6.64, 'E': -6.81,
                    'D': -8.72, 'R': -14.92}

    csv_labels = list(hydro_values.keys())

    def difference_matrix(values):
        '''
        compute pairwise differences for relational data
        :param values: float values to find differences between
        :return: matrix of differences (2D array)
        '''
        os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')
        vals = list(values.values())

        matrix = np.zeros((20, 20))

        for i in range(len(vals)):
            for j in range(len(vals)):
                matrix[i][j] = abs(vals[i] - vals[j]) / 20

        matrix = pd.DataFrame(data=matrix, index=csv_labels, columns=csv_labels)
        matrix['*'] = np.zeros((20))

        return matrix

    #compute matrix
    matrix = difference_matrix(hydro_values)

    matrix.to_csv(path_or_buf='protein_hphob.csv')

def float_to_rank():
    '''
    convert dictionary of protein hphob values to integers based on rank
    :return: NA, output to txt file
    '''
    hphob_values = {'I': 4.92, 'L': 4.92, 'V': 4.04, 'P': 2.98, 'F': 2.98, 'M': 2.35,
                    'W': 2.33, 'A': 1.81, 'C': 1.28, 'G': 0.94, 'Y': -0.14, 'T': -2.57,
                    'S': -3.40, 'H': -4.66, 'Q': -5.54, 'K': -5.55, 'N': -6.64, 'E': -6.81,
                    'D': -8.72, 'R': -14.92, '*': float('-inf')}


########---------actual module code--------------############


def load_data(filename, delim=','):
    '''
    import data as pandas dataframe
    :param filename: string name of the file
    :param delim: string delimiter
    :return: dataframe version of the given file
    useful functions to check state of data
    # print(data.head(10))
    # print('data types '+str(data.dtypes))
    '''
    """read in the csv of gene sequences and RNAseq expression values
       returns the data as a pandas dataframe"""
    data = pd.read_csv(filepath_or_buffer=filename, delimiter=delim)

    return data


def my_max(seqs):
    '''
    determine the longest sequence
    :param seqs: list of string sequences
    :return: integer length of the longest sequence
    '''
    max_string = ''
    max_len = len(max_string)

    for seq in seqs:
        if len(seq) > max_len:
            max_string = seq
            max_len = len(max_string)

    return max_len


def base_to_one_hot(data, encode_dict):
    '''
    one hot helper function
    :param data: datafrmae containing protein sequences as strings
    :param encode_dict: dataframe that can be used to convert sequence bases/residues to one hot vectors
    :return: list of one hot encodings
    '''
    seqs = data['protein_sequence'].tolist()
    d = encode_dict.to_dict('list')

    # max length of sequence
    my_max_1 = my_max(seqs)

    newcol = np.zeros((len(seqs), my_max_1, 21))

    for k, seq in enumerate(seqs):
        # padding check
        for i in range(len(seq)):
            one_hot = d[seq[i]]
            for j in range(0,21):
                newcol[k][i][j] = one_hot[j]


    return newcol


def encode_hphob(data):
    '''
    preprocess protein data for embedding, convert amino acid bases to integers based on hydrophobicity
    :param data: dataframe with protein sequences and expression values
    :return: the same dataframe with a new int array encoding of the proteins
    '''
    hphob_values = {'I': 4.92, 'L': 4.92, 'V': 4.04, 'P': 2.98, 'F': 2.98, 'M': 2.35,
                    'W': 2.33, 'A': 1.81, 'C': 1.28, 'G': 0.94, 'Y': -0.14, 'T': -2.57,
                    'S': -3.40, 'H': -4.66, 'Q': -5.54, 'K': -5.55, 'N': -6.64, 'E': -6.81,
                    'D': -8.72, 'R': -14.92, '*':0}

    protein_seqs = data['protein_sequence'].tolist()
    hphob_encoding = [[hphob_values[base] for base in seq ] for seq in protein_seqs]
    data['hphob_encode'] = pd.Series(hphob_encoding).values

    return data

def get_set(x_data, y_data, set):
    '''
    return the train, test or val subset of the x or y data
    :param x_data: dataframe of x data
    :param y_data: dataframe of y data
    :param set: string subset of data to extract
    :return: new dataframe ready for encoding, ids, sequences, expression values
    '''
    x_select = x_data.loc[x_data['group'] == set]

    new_df = pd.concat([x_select, y_data], axis=1, join='inner')

    return new_df

def extract_y(data, tissue):
    '''
    reformat dataframe values into usable np arrays
    :param data: dataframe of sequence data and expression values
    :return: np arrays of y data
    '''
    # slice out just one hot vectors and protein levels
    slice = data.loc[:, [tissue]]
    y = np.array(slice[tissue].values)

    return y


def main():
    os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')
    '''
    train/test synthetic data
    
    # how long is the sequence and how many are there... for synthetic data
    l = 400
    n = 10000
    synth = c.get_example('protein', n, l)
    heavy_As = c.get_example('heavy_As', n, l)
    encode_dict = load_data('box-data/protein_onehot.csv')

    synth_encoded = encode_o_h(synth, encode_dict)
    a_encoded = encode_o_h(heavy_As, encode_dict)
    
    synth_encoded.to_csv('synth.csv')
    a_encoded.to_csv('a_synth.csv')

    '''
    tissue = 'Protein_Leaf_Zone_3_Growth'

    # load the data from file
    x_data = pickle.load(open('X.pkl', 'rb'))
    y_data = pickle.load(open('y.pkl', 'rb'))
    encode_dict = load_data('protein_onehot.csv')

    train = get_set(x_data, y_data, 'train')
    val = get_set(x_data, y_data, 'val')
    test = get_set(x_data, y_data, 'test')

    train_encoded = base_to_one_hot(train, encode_dict)
    val_encoded = base_to_one_hot(val, encode_dict)
    test_encoded = base_to_one_hot(test, encode_dict)

    train = extract_y(train, tissue)
    test = extract_y(test, tissue)
    val = extract_y(val, tissue)

    return train_encoded, train, test_encoded, test, val_encoded, val


if __name__ == '__main__':
    main()
