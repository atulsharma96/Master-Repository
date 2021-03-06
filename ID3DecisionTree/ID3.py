import numpy as np
from collections import OrderedDict


class Node(object):
    def __init__(self, label, attrib, attrib_val):
        self.branches = list()
        self.label = label
        self.attrib = attrib
        self.attrib_val = attrib_val
        self.depth = 0


class ID3:
    def __init__(self, data_obj, attributes, label, depth=float("inf")):
        self.data_obj = data_obj
        self.attributes = attributes
        self.label = label
        self.depth = depth
        self.tree_root = None

    @staticmethod
    def max_depth(node):
        if node is None:
            return 0
        if node.attrib_val is None and node.attrib is None:
            return node.depth - 1
        if len(node.branches) == 0:
            return node.depth-1
        else:
            s = []
            for branch in node.branches:
                s.append(ID3.max_depth(branch))
            return max(s)

    @staticmethod
    def entropy(a):
        labeled_attrib = a.get_column('label')
        vals, counts = np.unique(labeled_attrib, return_counts=True)
        probabilities = []
        for count in counts:
            probabilities.append(count/sum(counts))
        probabilities = np.array(probabilities)
        return -1.0*(probabilities.dot(np.log(probabilities)))

    @staticmethod
    def calculate_information_gain(data_obj, attribute):
        unique_vals = data_obj.get_attribute_possible_vals(attribute)
        counts = list()
        for val in unique_vals:
            counts.append(len(data_obj.get_row_subset(attribute, val)))
        summation = 0.0
        for value, count in zip(unique_vals, counts):
            hsa = ID3.entropy(data_obj.get_row_subset(attribute, value))
            summation += count * hsa
        summation /= float(len(data_obj.get_column('label')))
        hs = ID3.entropy(data_obj)
        # print("Entropy of S ,", attribute, "=", hs-summation)
        return hs - summation

    @staticmethod
    def most_popular_label(data_obj):
        """
        :return: The most popular label for the data set at hand
        """
        label_count = OrderedDict()
        for label in data_obj.get_column('label'):
            if label in label_count.keys():
                label_count[label] += 1
            else:
                label_count[label] = 1
        list_of_labels = list(label_count.items())
        return list_of_labels[len(list_of_labels)-1][0]

    def build_tree(self, max_depth=float("inf")):
        self.tree_root = self.__build_tree_internal(self.data_obj, self.attributes, self.label, list(), max_depth, 0)
        return self.tree_root

    def __build_tree_internal(self, data_obj, attributes, label, ignore, depth, next_depth):
        """
        The algorithm used to build our tree.
        :param data_obj: The data set
        :param attributes: The attributes to look over
        :param label: The labels pertaining to data_obj
        :param ignore: The attributes to jump over
        :param depth: The maximum depth + 1
        :param next_depth: The depth at which the next node would belong
        :return: A node object that is the root of the entire decision tree
        """
        if data_obj is None:
            data_obj = self.data_obj
        if next_depth is not None and depth is not None and next_depth >= depth:
            root = Node(ID3.most_popular_label(data_obj), None, None)
            root.depth = depth
            return root
        if len(set(label)) == 1:
            root = Node(label[0], None, None)
            root.depth = next_depth
            return root
        else:
            root = Node(None, None, None)
            max_ig = float("-inf")
            best_attrib = None
            for a in attributes:
                info_gain = ID3.calculate_information_gain(data_obj, a)
                if info_gain > max_ig and str(best_attrib) != "label" and str(best_attrib) not in ignore:
                    max_ig = info_gain
                    best_attrib = a
            ignore.append(str(best_attrib))
            if best_attrib is None:
                print("Best attrib is none.")
            root.attrib = best_attrib
            for value in data_obj.get_attribute_possible_vals(best_attrib):
                new_branch = Node(None, best_attrib, value)
                sv = data_obj.get_row_subset(best_attrib, value)
                if len(sv) == 0:
                    length_of_labels = len(data_obj.get_column('label'))
                    if length_of_labels == 0:
                        continue
                    label_count = OrderedDict()
                    for label in data_obj.get_column('label'):
                        if label in label_count.keys():
                            label_count[label] += 1
                        else:
                            label_count[label] = 1
                    list_of_labels = list(label_count.items())
                    new_branch.label = list_of_labels[len(list_of_labels) - 1][0]
                    root.branches.append(new_branch)
                else:
                    new_attributes = dict()
                    for key in attributes.keys():
                        if key == str(best_attrib):
                            continue
                        else:
                            new_attributes[key] = attributes[key]
                    sv.attributes = new_attributes
                    if next_depth is None:
                        next_depth = 0
                    new_branch.branches.append(self.__build_tree_internal(sv, new_attributes, sv.get_column('label'), ignore, depth, (next_depth + 1)))

                    root.branches.append(new_branch)
            return root

    def find_label(self, point, tree_root):
        """
        :param point: The data row that we are trying to obtain the label for
        :param tree_root: The node to start looking for the label at
        :return: A single character string with the prediction for point
        """
        if tree_root.label is not None:
            return tree_root.label
        else:
            for branch in tree_root.branches:
                if branch.attrib is not None and branch.attrib_val is None:
                    return self.find_label(point, branch)
                elif branch.attrib is not None and branch.attrib_val is not None:
                    if point[self.data_obj.get_column_index(branch.attrib)] == branch.attrib_val:
                        return self.find_label(point, branch)
                    else:
                        continue
                else:
                    return self.find_label(point, branch)

    def predict(self, data_to_label):
        """

        This function predicts labels for the entire dataset.

        :param data_to_label: The dataset to predict labels for.
        :param tree: The root node with pointers to the entire tree
        :return: list with all the predicted labels
        """

        if self.tree_root is None:
            raise Exception("Cannot predict without building the tree first.")
        most_pop = self.most_popular_label(data_to_label)
        data_as_list = data_to_label.raw_data
        labels = list()
        for data_row in data_as_list:
            label_returned = self.find_label(data_row, self.tree_root)
            if label_returned is not None:
                labels.append(label_returned)
            else:
                labels.append(most_pop)
        return labels
