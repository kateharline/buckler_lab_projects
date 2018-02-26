import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import walley as w
import os
import platform

def check_families(fams):
    '''
    debugging empty families being appended
    :param fams: nested list of strings gene families
    :return: NA
    '''
    zero_fams = 0

    for i, fam in enumerate(fams):
        # print(str(i)+ ' | '+str(len(fam))+' '+str(fam))
        if len(fam) == 0:
            zero_fams += 1
    print('families with zero members '+str(zero_fams))


def get_avg_expression(families, exp_values, tissues):
    '''
    return the average expression level for each gene family
    :param families: list of lists where each family is a list of gene_ids
    :param exp_values: dictionary of gene id keys and expression values
    :return: np array of floats, avg expression for each family
    '''
    averages = np.zeros((len(tissues), len(families)))

    for i, tissue in enumerate(tissues):
        for j, family in enumerate(families):
            fam_sum = 0
            for gene in families[j]:
                fam_sum += exp_values[tissue][gene]
            averages[i][j] = fam_sum / len(family)

    return averages


def get_max_exp(families, exp_values, tissues):
    '''
    return the max expression level for each gene family
    :param families: list of lists where each family is a list of gene_ids
    :param exp_values: dictionary of gene id keys and expression values
    :return: np array of floats, max expression for each family
    '''
    maxs = np.zeros((len(tissues), len(families)))
    genes = np.zeros((len(tissues), len(families)), dtype=str)

    for i, tissue in enumerate(tissues):
        for j, family in enumerate(families):
            max_exp = 0
            for gene in families[j]:
                if exp_values[tissue][gene] > max_exp: max_exp = exp_values[tissue][gene]
            maxs[i][j] = max_exp
            genes[i][j] = gene

    return maxs, genes

def get_sizes(families):
    '''
    return the size of each family
    :param families: list of lists of families
    :return: np array of int lengths
    '''
    lengths = np.zeros((len(families)))

    for i, family in enumerate(families):
        lengths[i] = len(family)

    return lengths


def plot_exp_distribution(avgs, bins, type, mask=False):
    if mask:
        avgs_m = np.ma.masked_equal(avgs, 0)
        plt.hist(avgs_m, bins, facecolor='blue', alpha=0.5)
        plt.title('Expression Distribution Across Gene Families (M)')
    else:
        plt.hist(avgs, bins, facecolor='blue', alpha=0.5)
        plt.title('Expression Distribution Across Gene Families')
    plt.xlabel(str(type)+' Expression')
    plt.ylabel('Frequency Among Families')

    plt.show()
    return None

def plot_size_distribution(sizes, bins):
    plt.hist(sizes, bins, facecolor='blue', alpha=0.5)
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.title('Size of families')
    plt.show()
    return None

def plot_relationship(sizes, avgs, type, mask=False):
    if mask:
        avgs_m = np.ma.masked_equal(avgs, 0)
        plt.scatter(sizes, avgs_m)
        plt.title('Relationship Between Family Size and Expression (M)')
    else:
        plt.scatter(sizes, avgs)
        plt.title('Relationship Between Family Size and Expression')
    plt.xlabel('Size')
    plt.ylabel(str(type)+' Expression within Family')
    plt.show()
    return None

def pick_families(fams, avgs, sizes):
    '''
    pick families for overfitting
    :param fams: all gene families
    :return: families as input to the model
    '''
    selected_fams = []

    for i, family in enumerate(fams):
        if sizes[i] > 150 or avgs[i] > 8:
            selected_fams.append(family)

    return selected_fams


def pick_bal(n, p, genes, maxs):
    '''
    pick sample of total n genes with p split between high expression and low expression genes from families
    :param n: int total subset size
    :param p: float ratio of high expression to low expression
    :param genes: np array of strings names of genes with max expression from a family
    :return: list of string gene names to then be analyzes
    '''
    threshold = 2

    hi_genes = []
    lo_genes = []

    for i, gene in enumerate(genes):
        if maxs[i] > threshold:
            hi_genes.append(gene)
        else:
            lo_genes.append(gene)

    subset = pick_rand(n*p, hi_genes)
    subset += pick_rand(1-n*p, lo_genes)

    return subset

def pick_rand(n, genes):
    '''
    randomly pick genes for subset of size n
    :param n: int size
    :param genes: np array of str gene names
    :return: list of str gene names
    '''
    subset = np.random.choice(genes, n, replace=False)

    return subset

def main():
    if 'Ubuntu' in platform.platform():
        os.chdir('/home/kh694/Desktop/buckler-lab/box-data')
    else:
        os.chdir('/home/kateharline/Desktop/buckler-lab/box-data')

    selected_tissues = ['Leaf_Zone_3_Growth']

    v3_to_v4, genes = w.make_V3_converter('v3_v4_xref.txt')
    # select out genes not in complete dataset
    genes = w.load_promoter_data(genes, 'promoterSequences_v4.txt')[0]
    genes = w.load_us_data(genes, 'upstreamSequences_v4.txt')[0]
    genes = w.load_transcript_data(genes, 'transcriptSequences_v4.txt')[0]
    genes = w.load_transcript_level(genes, 'v4_Transcripts_meanLogExpression.csv')[0]

    # load actual data to model
    protein_DF, proteinSequence_DF, genes = w.load_protein_data(genes, 'Zea_mays.AGPv4.pep.longest.pkl', 'v4_Protein_meanLogExpression.csv')
    gene_families = w.define_families('gene_families.npy', 'nodes.npy', genes, v3_to_v4)
    # analyze info about gene families to make a subset
    avgs = get_avg_expression(gene_families, protein_DF[selected_tissues].to_dict(), selected_tissues)
    maxs, max_genes = get_max_exp(gene_families, protein_DF[selected_tissues].to_dict(), selected_tissues)
    sizes = get_sizes(gene_families)

    # plot_exp_distribution(avgs[0], 100)
    # plot_size_distribution(sizes, 100)
    # plot_relationship(sizes, avgs[0])
    # plot_exp_distribution(maxs[0][1], 100, 'Max', mask=True)
    # plot_exp_distribution(avgs[0], 100, 'Average', mask=True)
    # plot_relationship(sizes, avgs[0], 'Average', mask=True)
    # plot_relationship(sizes, maxs[0][1], 'Max')
    # plot_relationship(sizes, maxs[0][1], 'Max', mask=True)

    # make splits for different analyses

    smaller_fams = pick_families(gene_families, avgs[0], sizes) # 358 families, simple model
    # balanced_fams = pick_bal(200, 0.5, max_genes, maxs) # set to split for bal v unbal question
    # unbalanced_fams = pick_rand(200, max_genes)

    # convert subset to usable dataframe for making models
    genes_test, genes_train, genes_val = w.make_splits(smaller_fams)
    x, y = w.format_final_df(protein_DF, proteinSequence_DF, genes_train, genes_val, genes_test)

    return x, y


if __name__ == '__main__':
    main()
