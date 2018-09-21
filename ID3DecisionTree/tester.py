import ID3
from data import Data
import numpy as np
DATA_DIR = 'data_new/'

if __name__ == "__main__":
    print("Training on train.csv and testing on test.csv with no depth restriction...")
    ignore = list()
    data = np.loadtxt(DATA_DIR+'train.csv', delimiter=',', dtype=str)
    data_obj = Data(data=data)
    tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore)
    # tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, 7, data_obj, 0)
    # tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, 5, data_obj, 0)
    labels = ID3.predict(tree, data_obj)
    data = np.loadtxt(DATA_DIR+'test.csv', delimiter=',', dtype=str)
    data_obj = Data(data=data)
    labels = ID3.predict(tree, data_obj)
    labels_true = data_obj.get_column('label')
    print("Depth =", ID3.max_depth(tree))
    error = 0
    total = 0
    for l1, l2 in zip(labels, labels_true):
        total += 1
        if l1 != l2:
            error += 1
    error = float(error)/float(total) * 100
    print("Error =", error, "%")
