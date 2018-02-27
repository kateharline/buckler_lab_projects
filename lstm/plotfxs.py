import matplotlib.pyplot as plt

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
