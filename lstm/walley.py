import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle

# Selected tissue
selected_tissues = ['Leaf_Zone_3_Growth', 'Root_Meristem_Zone_5_Days']

# File locations
os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')

v3_to_v4_file = 'v3_v4_xref.txt'
proteinSequence_file = 'Zea_mays.AGPv4.pep.longest.pkl'
index_family_file = 'gene_families.npy'
node_file = 'nodes.npy'

proteinLevel_file = 'v4_Protein_meanLogExpression.csv'

#########################################################
# Functions
#########################################################


def unlist(list_object):
    return [element for sublist in list_object for element in sublist]


#########################################################
# Data
#########################################################
def make_V3_converter(v3_to_v4_file):
    # v3 to v4
    v3_to_v4_DF = pd.read_table(v3_to_v4_file)
    v3_to_v4_DF.index = v3_to_v4_DF['v3_gene_model']
    v3_to_v4 = v3_to_v4_DF['v4_gene_model'].to_dict()

    genes = set(v3_to_v4.values())

    return v3_to_v4, genes

def load_promoter_data(genes, promoterSequence_file):
    # Promoter information
    promoterSequence_DF = pd.read_table(promoterSequence_file)
    promoterSequence_DF.index = promoterSequence_DF['gene_id']

    genes1 = genes.intersection(set(promoterSequence_DF.index))
    return genes1, promoterSequence_DF

def load_us_data(genes, upstreamSequence_file):
    upstreamSequence_DF = pd.read_table(upstreamSequence_file)
    upstreamSequence_DF.index = upstreamSequence_DF['gene_id']

    genes1 = genes.intersection(set(upstreamSequence_DF.index))
    return genes1, upstreamSequence_DF

def load_transcript_data(genes, transcriptSequence_file):
    transcriptSequences = pd.read_table(transcriptSequence_file)
    transcriptSequences['gene_id'] = [tx_name.split('_')[0] for tx_name in transcriptSequences['tx_name']]

    transcriptSequence_DF = transcriptSequences.sort_values(['width'], ascending=False).groupby(['gene_id']).head(1)
    transcriptSequence_DF.index = transcriptSequence_DF['gene_id']

    genes1 = genes.intersection(set(transcriptSequence_DF.index))

    return genes1, transcriptSequence_DF

def load_transcript_level(genes, transcriptLevel_file):
    transcriptLevel_DF = pd.read_csv(transcriptLevel_file, dtype={'v4_geneIDs':str})
    transcriptLevel_DF = transcriptLevel_DF[transcriptLevel_DF['duplicated_v4'] != True]
    transcriptLevel_DF = transcriptLevel_DF[transcriptLevel_DF['v4_geneIDs'] != 'None']
    transcriptLevel_DF.index = transcriptLevel_DF['v4_geneIDs']

    genes1 = genes.intersection(set(transcriptLevel_DF.index))

    return genes1, transcriptLevel_DF


def load_protein_data(genes, proteinSequence_file, proteinLevel_file):
    # Protein sequence information
    proteinSequences = pickle.load(open(proteinSequence_file, 'rb'))
    proteinSequence_DF = pd.DataFrame({'sequence': list(proteinSequences.values())},
                                      index=proteinSequences.keys())

    genes1 = genes.intersection(set(proteinSequence_DF.index))

    # Protein levels
    proteinLevel_DF = pd.read_csv(proteinLevel_file)
    proteinLevel_DF = proteinLevel_DF[proteinLevel_DF['duplicated_v4'] != True]
    proteinLevel_DF = proteinLevel_DF[proteinLevel_DF['v4_geneIDs'] != 'None']
    proteinLevel_DF.index = proteinLevel_DF['v4_geneIDs']

    genes1 = genes1.intersection(set(proteinLevel_DF.index))

    return proteinLevel_DF, proteinSequence_DF, genes1

#########################################################
# Gene families
#########################################################
# Gene family memberships
def remove_empties(fams):
    '''
    remove the empty lists that are being added to the list for some reason... if statement?
    :param fams: list of lists with some empty lists
    :return: list of lists without the empty lists
    '''
    no_empties = []
    for fam in fams:
        if len(fam) > 0:
            no_empties.append(fam)

    return no_empties

def define_families(index_family_file, node_file, genes, v3_to_v4):

    index_families = np.load(index_family_file)
    gene_nodes = np.load(node_file)

    gene_families = [[ v3_to_v4[gene_nodes[i]] for i in tuple  if gene_nodes[i] in v3_to_v4.keys() and v3_to_v4[gene_nodes[i]] in genes] for tuple in index_families ]
    gene_families = remove_empties(gene_families)
    # Add singletons to gene families
    in_family = set([gene for gene_tuple in gene_families for gene in gene_tuple])
    singletons = set(genes) - in_family

    gene_families += [[singleton] for singleton in singletons]
    return gene_families

# Splits
def make_splits(gene_families):

    families_cal, families_test = train_test_split(gene_families, test_size=0.15, random_state=0)
    families_train, families_val = train_test_split(families_cal, test_size=0.20, random_state=1)

    genes_train = unlist(families_train)
    genes_val = unlist(families_val)
    genes_test = unlist(families_test)

    return genes_test, genes_train, genes_val

#########################################################
# Input
#########################################################
# Gene groups
def format_final_df(proteinLevel_DF, proteinSequence_DF, genes_train, genes_val, genes_test, name=''):
    # select smaller subset
    genes = genes_train + genes_val + genes_test

    template = pd.DataFrame({'group': 'train'}, index=genes)
    template['group'][[gene in genes_val for gene in genes]] = 'val'
    template['group'][[gene in genes_test for gene in genes]] = 'test'

    # X
    X = pd.concat([
        template,
        proteinSequence_DF.loc[genes, 'sequence']
    ], axis=1)
    X.columns = ['group', 'protein_sequence']
    X.index.name = 'gene_id'

    # y
    y = pd.concat([
        template,
        proteinLevel_DF.loc[genes, selected_tissues]
    ], axis=1)
    y.columns = unlist([
        ['group'],
        ['Protein_'+tissue for tissue in selected_tissues]])
    y.index.name = 'gene_id'

    # Saving
    X.to_csv(str(name)+'X.csv')
    pickle.dump(X, open(str(name)+'X.pkl', 'wb'))

    y.to_csv(str(name)+'Y.csv')
    pickle.dump(y, open(str(name)+'Y.pkl', 'wb'))

    return X, y