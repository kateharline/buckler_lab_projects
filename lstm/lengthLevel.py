import matplotlib.pyplot as plt
import os
import pandas as pd

import walley as w

os.chdir('/Users/kateharline/Desktop/buckler-lab/protein_ml/data')

selected_tissues = ['Leaf_Zone_3_Growth']

v3_to_v4, genes = w.make_V3_converter('v3_v4_xref.txt')

genes = w.load_promoter_data(genes, 'promoterSequences_v4.txt')[0]
genes = w.load_us_data(genes, 'upstreamSequences_v4.txt')[0]
genes = w.load_transcript_data(genes, 'transcriptSequences_v4.txt')[0]
genes = w.load_transcript_level(genes, 'v4_Transcripts_meanLogExpression.csv')[0]

# load actual data to model
protein_DF, proteinSequence_DF, genes = w.load_protein_data(genes, 'Zea_mays.AGPv4.pep.longest.pkl', 'v4_Protein_meanLogExpression.csv')
gene_families = w.define_families('gene_families.npy', 'nodes.npy', genes, v3_to_v4)

os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data/lengthLevel')

genes_test, genes_train, genes_val = w.make_splits(gene_families)
x, y = w.format_final_df(protein_DF, proteinSequence_DF, genes_train, genes_val, genes_test)



both = pd.concat([x, y], axis=1, join='inner')

seqs = both[['protein_sequence']].to_dict('list')
levels = both[['Protein_Leaf_Zone_3_Growth']].to_dict('list')

x_plot = [ len(seq) for seq in seqs['protein_sequence'] ]
y_plot = levels['Protein_Leaf_Zone_3_Growth']



plt.scatter(x_plot, y_plot)
plt.title('Do smaller proteins have higher abundance?')
plt.xlabel('Protein Length')
plt.ylabel('Protein Level')
plt.show()
