import numpy as np
from data import Data
import ID3
if __name__ == "__main__":
    DATA_DIR = 'data_new/CVfolds_new/'
    print("Running cross validation experiment...")
    depth_values = [1, 2, 3, 4, 5, 10, 15]
    error_vals = [0.0] * len(depth_values)
    standard_deviation = [0.0] * len(depth_values)
    for depth in range(0, len(depth_values)):
        print("Running with depth:", depth_values[depth])
        error_local = [0.0]*5
        for k in range(1, 6):
            if k == 1:
                array = np.loadtxt(DATA_DIR+"fold2.csv", delimiter=",", dtype=str)
                array = np.concatenate((array, np.loadtxt(DATA_DIR+"fold3.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR+"fold4.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR+"fold5.csv", delimiter=",", dtype=str)[1:]))
                data_obj = Data(data=array)
                ignore = list()
                tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth_values[depth]+1, data_obj, 0)
                evaluator = np.loadtxt(DATA_DIR+"fold1.csv", delimiter=",", dtype=str)
                evaluator = Data(data=evaluator)
                labels = ID3.predict(tree, evaluator)
                labels_true = evaluator.get_column('label')
                total = 0
                error = 0
                for l1, l2 in zip(labels, labels_true):
                    total += 1
                    if l1 != l2:
                        error += 1
                error = float(error) / float(total)
                error_local[k-1] = error
                error_vals[depth] += error
            if k == 2:
                array = np.loadtxt(DATA_DIR + "fold1.csv", delimiter=",", dtype=str)
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold3.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold4.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold5.csv", delimiter=",", dtype=str)[1:]))
                data_obj = Data(data=array)
                ignore = list()
                tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth_values[depth]+1,
                                      data_obj, 0)
                evaluator = np.loadtxt(DATA_DIR + "fold2.csv", delimiter=",", dtype=str)
                evaluator = Data(data=evaluator)
                labels = ID3.predict(tree, evaluator)
                labels_true = evaluator.get_column('label')
                total = 0
                error = 0
                for l1, l2 in zip(labels, labels_true):
                    total += 1
                    if l1 != l2:
                        error += 1
                error = float(error) / float(total)
                error_local[k - 1] = error
                error_vals[depth] += error
            if k == 3:
                array = np.loadtxt(DATA_DIR + "fold2.csv", delimiter=",", dtype=str)
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold1.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold4.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold5.csv", delimiter=",", dtype=str)[1:]))
                data_obj = Data(data=array)
                ignore = list()
                tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth_values[depth]+1,
                                      data_obj, 0)
                evaluator = np.loadtxt(DATA_DIR + "fold3.csv", delimiter=",", dtype=str)
                evaluator = Data(data=evaluator)
                labels = ID3.predict(tree, evaluator)
                labels_true = evaluator.get_column('label')
                total = 0
                error = 0
                for l1, l2 in zip(labels, labels_true):
                    total += 1
                    if l1 != l2:
                        error += 1
                error = float(error) / float(total)
                error_local[k - 1] = error
                error_vals[depth] += error
            if k == 4:
                array = np.loadtxt(DATA_DIR + "fold2.csv", delimiter=",", dtype=str)
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold3.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold1.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold5.csv", delimiter=",", dtype=str)[1:]))
                data_obj = Data(data=array)
                ignore = list()
                tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth_values[depth]+1,
                                      data_obj, 0)
                evaluator = np.loadtxt(DATA_DIR + "fold4.csv", delimiter=",", dtype=str)
                evaluator = Data(data=evaluator)
                labels = ID3.predict(tree, evaluator)
                labels_true = evaluator.get_column('label')
                total = 0
                error = 0
                for l1, l2 in zip(labels, labels_true):
                    total += 1
                    if l1 != l2:
                        error += 1
                error = float(error) / float(total)
                error_local.append(error)
                error_vals[depth] += error
            if k == 5:
                array = np.loadtxt(DATA_DIR + "fold2.csv", delimiter=",", dtype=str)
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold3.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold4.csv", delimiter=",", dtype=str)[1:]))
                array = np.concatenate((array, np.loadtxt(DATA_DIR + "fold1.csv", delimiter=",", dtype=str)[1:]))
                data_obj = Data(data=array)
                ignore = list()
                tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth_values[depth]+1,
                                      data_obj, 0)
                evaluator = np.loadtxt(DATA_DIR + "fold5.csv", delimiter=",", dtype=str)
                evaluator = Data(data=evaluator)
                labels = ID3.predict(tree, evaluator)
                labels_true = evaluator.get_column('label')
                total = 0
                error = 0
                for l1, l2 in zip(labels, labels_true):
                    total += 1
                    if l1 != l2:
                        error += 1
                error = float(error) / float(total)
                error_local[k - 1] = error
                error_vals[depth] += error
        error_local = np.array(error_local)
        standard_deviation[depth] = np.std(error_local)
    for i in range(0, len(error_vals)):
        error_vals[i] /= 5.0
    print("Errors and S.D. corresponding to 1, 2, 3, 4, 5, 10 and 15 respectively.")
    print("Errors:", error_vals)
    print("Standard Deviations:", standard_deviation)
    depth_index = error_vals.index(min(error_vals))
    depth = depth_values[depth_index]
    data = np.loadtxt('data_new/train.csv', delimiter=',', dtype=str)
    data_obj = Data(data=data)
    ignore = list()
    print("Training with train.csv with depth limited to ", depth)
    tree = ID3.build_tree(data_obj, data_obj.attributes, data_obj.get_column('label'), ignore, depth+1, data_obj, 0)
    data = np.loadtxt('data_new/test.csv', delimiter=',', dtype=str)
    data_obj = Data(data=data)
    labels = ID3.predict(tree, data_obj)
    labels_true = data_obj.get_column('label')
    total = 0
    error = 0
    for l1, l2 in zip(labels, labels_true):
        total += 1
        if l1 != l2:
            error += 1
    error = float(error) / float(total)
    print("Accuracy of predictions on test.csv:", (1.0-error)*100, "%")
