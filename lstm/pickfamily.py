import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py

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


def plot_size_distribution(avgs):

    plt.hist(avgs)
    plt.xlabel('Average Expression')
    plt.ylabel('Frequency Among Families')
    plt.title('Expression Distribution Across Gene Families')
    fig = plt.gcf()

    plot_url = py.plot_mpl(fig, filename='family-exp')

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

