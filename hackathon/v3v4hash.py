import pandas as pd

def load(filepath, separator):
    """read in the csv of gene sequences and RNAseq expression values
       returns the data as a pandas dataframe"""
    data = pd.read_csv(filepath_or_buffer=filepath, sep=separator)
    # print(data.head(10))
    # print('data types '+str(data.dtypes))
    return data


def hash(conv, input, v3, v4):
    """ convert the v3 labels from the input to v4 by indexing into the v3 to v4 data"""
    data = conv[[v3][v4]]

    data = data.to_dict('split')
    # the index of the row that holds the translation
    index_v3 = data['columns'].index(v3)
    index_v4 = data['columns'].index(v4)

    holding = []

    print('data check '+str(input.head(10)))

    for row in input.itertuples():
        # get gene name from expression data
        v3_name = row[1]
        # find that name in the conversion table
        new_name = data[]

    return ''

def main():
    conversion_file_path = 'v3_v4_xref.txt'
    input_file_path = 'RNA_expression_meanLogExpression.csv'
    output_file_path = 'rna_meanLogExpression'

    conv = load(conversion_file_path, '\t')
    input = load(input_file_path, ',')

    v3_name = 'v3_gene_model'
    v4_name = 'v4_gene_model'


    hash_Result = hash(conv, input, v3_name, v4_name)


if __name__ == '__main__':
    main()

print('Hello world')
