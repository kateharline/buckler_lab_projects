'''create a data set of known rules to "train" models on
current rules: A content in protein seq is related to protein expression level

'''

import numpy as np
import random

def make_xs(n, l, t):
    '''randomly generate n number of of type t sequences of length l'''
    seqs = []
    for i in range(n):
        s = ''.join(random.choice(t) for _ in range(l))
        seqs.append(s)

    return seqs


def make_ys(seqs):
    '''translate random sequences into expression values based on rules above'''

    return None


def main():
    l = 400
    n = 4

    dna_xs = make_xs(n, l, ['A', 'C', 'G', 'T'])
    protein_xs = make_xs(n, l, ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    dna_ys = make_ys(dna_xs)
    protein_ys = make_ys(protein_xs)

if __name__ == '__main__':
    main()