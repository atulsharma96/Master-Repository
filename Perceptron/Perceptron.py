import numpy as np
import random


class TrainingSetNotInitializedException(Exception):
    def __init__(self, message):
        super(TrainingSetNotInitializedException, self).__init__(message)


class Perceptron:
    def __init__(self, num_of_features=19):
        self.num_updates = 0.0
        self.x = []
        self.labels = []

        rand_val = 0.0
        # fill vector with non-zero values
        while rand_val == 0.0:
            rand_val = random.randrange(-10, 10)
        self.w = np.full((num_of_features, 1), rand_val / 1000.0)

        self.a = np.zeros((num_of_features, 1))
        self.b = rand_val/1000.0
        self.b_a = 0.0
        self.mode = None
        self.num_of_features = num_of_features

    def read_file(self, path, append=False):
        """
        Reads in specified file into the model and uses it as training data when the train function is run.
        :param path: The path to the file to read in as training data.
        :return: None
        """
        data_file = open(path)
        example_list = list()
        labels = list()
        for line in data_file:
            line_vals = line.split(' ')
            labels.append(int(line_vals[0]))
            features = np.zeros((self.num_of_features, 1))
            for i in range(0, len(line_vals)):
                if i == 0:
                    continue
                index, value = line_vals[i].split(":")
                features[int(index)-1] = float(value)
            example_list.append(features)

        if append:
            self.labels = np.array([l for l in self.labels] + labels)
            self.x = np.array([x for x in self.x] + example_list)
        else:
            self.labels = np.array(labels)
            self.x = np.array(example_list)

    def train(self, r_0=1.0, mode='std', mu=1.0, t=10):  # t is the maximum training epoch
        """
        Function to train the Perceptron algorithm.
        :param r_0: Initial learning rate (float)
        :param mode: Can be 'std' for standard, 'average' for average, 'decay' for decay and 'margin' for margin. Default is 'std'. (string)
        :param mu: Must be specified for the margin mode. Default value is 1.0 (float)
        :param t: The number of epochs that we want to train our algorithm for. (int)
        :return: The weight vector you can use to make your own predictions. (numpy.array)
        """
        self.mode = mode
        if self.x is None:
            e = TrainingSetNotInitializedException("Must read in a training set using read_file before using this function.")
            raise e

        r = r_0
        for T in range(0, t):

            # shuffle data in tandem with the labels
            rng_state = np.random.get_state()
            np.random.shuffle(self.x)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
            # end shuffle

            count = 0
            if mode == 'decay':
                for example in self.x:
                    prod = np.dot(np.transpose(self.w), example)+self.b
                    prod *= self.labels[count]
                    if prod <= 0:
                        self.num_updates += 1.0
                        self.w = self.w + r * (self.labels[count] * example)
                        self.b = self.b + r * self.labels[count]#self.labels[count - 1]
                    count += 1
                    r = r_0/float((T+1))
                # return self.w
            elif mode == 'margin':
                for example in self.x:
                    prod = np.dot(np.transpose(self.w), example) + self.b
                    prod *= self.labels[count]
                    if prod <= 0 or prod <= mu:
                        self.num_updates += 1.0
                        self.w = self.w + r * (self.labels[count] * example)
                        self.b = self.b + r * self.labels[count]#self.labels[count - 1]
                    count += 1
                    r = r_0 / float((T+1))
                # return self.w
            elif mode == 'average':
                for example in self.x:
                    prod = np.dot(np.transpose(self.w), example) + self.b
                    prod *= self.labels[count]
                    if prod <= 0:
                        self.num_updates += 1.0
                        self.w = self.w + r*(self.labels[count]*example)
                        self.b = self.b + r * self.labels[count]#self.labels[count - 1]
                    count += 1
                    self.a = self.a+self.w
                    self.b_a += self.b
                # return self.a
            else:
                for example in self.x:
                    prod = np.dot(np.transpose(self.w), example) + self.b
                    prod *= self.labels[count]
                    if prod <= 0:
                        self.num_updates += 1.0
                        self.w = self.w + r * (self.labels[count] * example)
                        self.b = self.b + r * self.labels[count]#self.labels[count - 1]
                    count += 1
                # return self.w
        if mode == 'average':
            return self.a
        return self.w

    def predict(self, data):
        """
        Returns the predictions for given data.
        :param data: The numpy array consisting of examples to predict labels for. (numpy.array)
        :return: A list consisting of the labels ordered by data. (list)
        """
        if self.x is None:
            raise TrainingSetNotInitializedException("Cannot predict label without reading in training set using"
                                                     "read_file and training dataset using train.")
        label = list()
        for sample in data:
            if not self.mode == 'average':
                prod = np.dot(np.transpose(self.w), sample) + self.b
            else:
                prod = np.dot(np.transpose(self.a), sample) + self.b_a
            if prod >= 0:
                y = 1
            else:
                y = -1
            label.append(y)
        return label
