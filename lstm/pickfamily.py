import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import walley as w
import os

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


def plot_exp_distribution(avgs, bins):
    plt.hist(avgs, bins, facecolor='blue', alpha=0.5)
    plt.xlabel('Average Expression')
    plt.ylabel('Frequency Among Families')
    plt.title('Expression Distribution Across Gene Families')
    plt.show()
    return None

def plot_size_distribution(sizes, bins):
    plt.hist(sizes, bins, facecolor='blue', alpha=0.5)
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.title('Size of families')
    plt.show()
    return None

def plot_relationship(sizes, avgs):
    plt.scatter(sizes, avgs)
    plt.xlabel('Size')
    plt.ylabel('Average Expression within Family')
    plt.title('Realtionship Between Family Size and Expression')
    plt.show()
    return None

def pick_families(fams):
    '''
    
    :param fams:
    :return:
    '''
    selected_fams = []

    return selected_fams

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

def main():
    os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')

    selected_tissues = ['Leaf_Zone_3_Growth']

    v3_to_v4, genes = w.make_V3_converter('v3_v4_xref.txt')
    # select out genes not in complete dataset
    genes = w.load_promoter_data(genes, 'promoterSequences_v4.txt')[0]
    genes = w.load_us_data(genes, 'upstreamSequences_v4.txt')[0]
    genes = w.load_transcript_data(genes, 'transcriptSequences_v4.txt')[0]
    genes = w.load_transcript_level(genes, 'v4_Transcripts_meanLogExpression.csv')[0]

    # load actual data to model
    protein_DF, genes = w.load_protein_data(genes, 'Zea_mays.AGPv4.pep.longest.pkl', 'v4_Protein_meanLogExpression.csv')
    gene_families = w.define_families('gene_families.npy', 'nodes.npy', genes, v3_to_v4)
    protein_DF_selected = protein_DF[selected_tissues]

    # start here, formatting of protein dataframe, feeding in as list (probably want helper function)
    avgs = get_avg_expression(gene_families, protein_DF[selected_tissues].to_dict(), selected_tissues)

    sizes = get_sizes(gene_families)

    plot_exp_distribution(avgs[0], 10)
    plot_size_distribution(sizes, 10)
    plot_relationship(sizes, avgs[0])


if __name__ == '__main__':
    main()
