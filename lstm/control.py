'''create a data set of known rules to "train" models on
current rules: A content in protein seq is related to protein expression level

'''

import numpy as np
import random
import pandas as pd

def make_xs(n, l, t, w=None):
    '''randomly generate n number of of type t sequences of length l'''
    seqs = []
    for i in range(n):
        # s = ''.join(random.choices(t) for _ in range(l))
        s = ''.join(random.choices(t, weights=w, k=random.choices(range(50, l))[0]))
        seqs.append(s)

    return seqs


def make_ys(seqs, rules):
    '''translate random sequences into expression values based on rules above'''
    scores = []

    for seq in seqs:
        for rule in rules:
            scores.append(rule(seq))

    return scores

def rule_1(seq):
    a_count = seq.count('A')
    score = a_count*10 / len(seq)

    return score

def rule_2(seq):
    a_count = seq.count('A')
    score = a_count*2 / len(seq)

    return score

def make_df(x, x_title, y, y_title):

    df = pd.DataFrame(data={x_title : x, y_title : y})

    return df

def get_example(type, n, l):

    if type == 'protein':
        protein_xs = make_xs(n, l, ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*'])
        protein_ys = make_ys(protein_xs, [rule_1])
        protein_df = make_df(protein_xs, 'protein_seqs', protein_ys, 'p_levels')
        return protein_df


    if type == 'dna':
        dna_xs = make_xs(n, l, ['A', 'C', 'G', 'T'])
        dna_ys = make_ys(dna_xs, [rule_2])
        dna_df = make_df(dna_xs, 'dna_seq', dna_ys, 'p_levels')
        return dna_df

    if type == 'rna':
        rna_xs = make_xs(n, l, ['A', 'C', 'G', 'U'])
        rna_ys = make_ys(rna_xs, [rule_2])
        rna_df = make_df(rna_xs, 'rna_seq', rna_ys, 'p_levels')
        return rna_df

    if type == 'heavy_As':
        protein_xs = make_xs(n, l, ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], w=[.5, .025,
                                                                                            .025,.025,.025,.025,.025,
                                                                                            .025,.025,.025,.025,.025,.025,
                                                                                            .025,.025,.025,.025,.025,.025,
                                                                                            .025])
        protein_ys = make_ys(protein_xs, [rule_1])
        protein_df = make_df(protein_xs, 'proteins', protein_ys, 'p_levels')
        return protein_df


    else:
        return 'Please specify example type'



''' example usage 

l = 400
n = 4

dna_xs = make_xs(n, l, ['A', 'C', 'G', 'T'])
protein_xs = make_xs(n, l, ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
dna_ys = make_ys(dna_xs, [rule_2])
protein_ys = make_ys(protein_xs, [rule_1])

print('dna_ys '+str(dna_ys))
print('protein_ys '+str(protein_ys))

'''