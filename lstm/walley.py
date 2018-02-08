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
# v3 to v4
v3_to_v4_DF = pd.read_table(v3_to_v4_file)
v3_to_v4_DF.index = v3_to_v4_DF['v3_gene_model']
v3_to_v4 = v3_to_v4_DF['v4_gene_model'].to_dict()

genes = set(v3_to_v4.values())

# Protein sequence information
proteinSequences = pickle.load(open(proteinSequence_file, 'rb'))
proteinSequence_DF = pd.DataFrame({'sequence': list(proteinSequences.values())},
                                  index=proteinSequences.keys())

genes = genes.intersection(set(proteinSequence_DF.index))

# Protein levels
proteinLevel_DF = pd.read_csv(proteinLevel_file)
proteinLevel_DF = proteinLevel_DF[proteinLevel_DF['duplicated_v4'] != True]
proteinLevel_DF = proteinLevel_DF[proteinLevel_DF['v4_geneIDs'] != 'None']
proteinLevel_DF.index = proteinLevel_DF['v4_geneIDs']

genes = genes.intersection(set(proteinLevel_DF.index))

#########################################################
# Gene families
#########################################################
# Gene family memberships
index_families = np.load(index_family_file)

gene_nodes = np.load(node_file)

gene_families = [[v3_to_v4[gene_nodes[i]] for i in gene_tuple if gene_nodes[i] in genes] for gene_tuple in index_families]

# Add singletons to gene families
in_family = set([gene for gene_tuple in gene_families for gene in gene_tuple])
singletons = set(genes) - in_family

gene_families += [[singleton] for singleton in singletons]

# Splits
families_cal, families_test = train_test_split(gene_families, test_size=0.15, random_state=0) # ----change this line to choose
    # most different families
families_train, families_val = train_test_split(families_cal, test_size=0.20, random_state=1)

genes_train = unlist(families_train)
genes_val = unlist(families_val)
genes_test = unlist(families_test)

#########################################################
# Input
#########################################################
# Gene groups
template = pd.DataFrame({'group': 'train'}, index=genes)
template['group'][[gene in genes_val for gene in genes]] = 'val'
template['group'][[gene in genes_test for gene in genes]] = 'test'

# X
X = pd.concat([
    template,
    proteinSequence_DF.loc[genes, 'sequence']
], axis=1)
X.columns = ['group', 'TSS_promoter', 'ATG_promoter', 'transcript_sequence', 'protein_sequence']
X.index.name = 'gene_id'

# y
y = pd.concat([
    template,
    proteinLevel_DF.loc[genes, selected_tissues]
], axis=1)
y.columns = unlist([
    ['group'],
    ['RNA_'+tissue for tissue in selected_tissues],
    ['Protein_'+tissue for tissue in selected_tissues]])
y.index.name = 'gene_id'

# Saving
X.to_csv('X.csv')
pickle.dump(X, open('X.pkl', 'wb'))

y.to_csv('y.csv')
pickle.dump(y, open('y.pkl', 'wb'))

# # Promoter sequences
# promoterSequences = list(promoterSequence_DF.loc[genes, 'sequence'])
#
# base2vec = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
# X_promoter = {index: [base2vec[base] for base in list(seq)]
#               for index, seq in zip(promoterSequence_DF['gene_id'], promoterSequence_DF['sequence'])}
#
# X_promoter_train = {gene: X_promoter[gene] for gene in genes_train}
# X_promoter_val = {gene: X_promoter[gene] for gene in genes_val}
# X_promoter_test = {gene: X_promoter[gene] for gene in genes_test}
