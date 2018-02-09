import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import walley as w
import os

def get_avg_expression(families, exp_values):
    '''
    return the average expression level for each gene family
    :param families: list of lists where each family is a list of gene_ids
    :param exp_values: dictionary of gene id keys and expression values
    :return: np array of floats, avg expression for each family
    '''
    averages = np.zeros((len(families)))

    for i, family in enumerate(families):
        fam_sum = 0
        for gene in family:
            fam_sum += exp_values[gene]
        averages[i] = fam_sum / len(family)

    return averages


def get_sizes(families):
    '''
    return the size of each family
    :param families: list of lists of families
    :return: np array of int lengths
    '''
    lengths = np.zeros((len(families)))

    for i, family in enumerate(families):
        lengths[i] = len(family)

    print('Average family size is '+str(np.mean(lengths)))

    return lengths


def plot_exp_distribution(avgs):

    plt.hist(avgs)
    plt.xlabel('Average Expression')
    plt.ylabel('Frequency Among Families')
    plt.title('Expression Distribution Across Gene Families')
    plt.show()

    return None

def plot_size_distribution(sizes):

    plt.hist(sizes)
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.title('Size of families')
    plt.show()

    return None

def plot_relationship(sizes, avgs):

    plt.plot(sizes, avgs)
    plt.xlabel('Size')
    plt.ylabel('Average Expression within Family')
    plt.title('Realtionship Between Family Size and Expression')
    plt.show()

    return None

def optimize(avgs, sizes, num_indices):
    '''
    find the optimal families to use with the goal of maximizing family size and variance between family averages
    :param avgs: array of average expression levels by family
    :param sizes: array of family sizes
    :param num_indices: number of families you want to return
    :return: array of indices of families
    '''
    family_indices = np.zeros((num_indices))


    return family_indices

def main():
    os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')

    selected_tissues = ['Leaf_Zone_3_Growth']

    v3_to_v4, genes = w.make_V3_converter('v3_v4_xref.txt')
    protein_DF, genes = w.load_protein_data(genes, 'Zea_mays.AGPv4.pep.longest.pkl', 'v4_Protein_meanLogExpression.csv')
    gene_families = w.define_families('gene_families.npy', 'nodes.npy', genes, v3_to_v4)
    protein_DF_selected = protein_DF[selected_tissues]

    '''
    # start here, formatting of protein dataframe, feeding in as list (probably want helper function)
    avgs = get_avg_expression(gene_families, protein_DF[selected_tissues].tolist())
    sizes = get_sizes(gene_families)

    plot_exp_distribution(avgs)
    plot_size_distribution(sizes)
    plot_relationship(sizes, avgs)
    '''

if __name__ == '__main__':
    main()
