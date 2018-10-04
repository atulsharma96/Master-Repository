from Perceptron import Perceptron
import numpy as np
# import matplotlib.pyplot as plt


def read_file(path):
    labels = list()
    data_file = open(path)
    example_list = list()
    for line in data_file:
        line_vals = line.split(' ')
        labels.append(int(line_vals[0]))
        features = np.zeros((19, 1))
        for i in range(0, len(line_vals)):
            if i == 0:
                continue
            index, value = line_vals[i].split(":")
            features[int(index) - 1] = float(value)
        example_list.append(features)
    return np.array(example_list), labels


def run_cross_validation(p, opath="/", mode="std", mu=0.0, rate=1.0):
    cv = [0.0]*5
    for i in range(0, 5):
        test = list()
        labels = list()
        for j in range(0, 5):
            file_to_read = "training0"+str(j)+".data"
            path = opath+file_to_read
            if i == j:
                file_in = read_file(path)
                test = file_in[0]
                labels = file_in[1]
                continue
            p.read_file(path)
            p.train(mode=mode, mu=mu, r_0=rate, t=10)
        predictions = p.predict(test)
        correct = 0.0
        total = 0.0
        for prediction in range(0,len(predictions)):
            if predictions[prediction] == labels[prediction]:
                correct += 1.0
            total += 1.0
        cv[i] = (correct/total)
    return np.array(cv)


if __name__ == "__main__":
    learning_rates = {1.0, 0.1, 0.01}
    margin_values = {1.0, 0.1, 0.01}
    modes = ['std', 'decay', 'margin', 'average']
    p_cv = Perceptron()
    best_mu = {'std': -1.0, 'decay': -1.0, 'margin': -1.0, 'average': -1.0}
    best_rate = {'std': -1.0, 'decay': -1.0, 'margin': -1.0, 'average': -1.0}
    accuracy = {'std': -1.0, 'decay': -1.0, 'margin': -1.0, 'average': -1.0}
    for mode in modes:
        for mu in margin_values:
            for rate in learning_rates:
                cv_acc = run_cross_validation(p_cv, 'dataset/CVSplits/', mode=mode, mu=mu, rate=rate)
                avg_acc = sum(cv_acc)/float(len(cv_acc))
                if avg_acc > accuracy[mode]:
                    accuracy[mode] = avg_acc
                    best_mu[mode] = mu
                    best_rate[mode] = rate

    for mode in modes:
        print("Best accuracy for", mode, "=", accuracy[mode])
        if mode == 'margin':
            print("Best mu for", mode, "=", best_mu[mode])
        print("Best learning rate/initial learning rate for", mode, "=", best_rate[mode])
        print("##########################################################################")

    best_epoch = {'std': 0, 'decay': 0, 'margin': 0, 'average': 0}

    data = read_file('dataset/diabetes.dev')
    number_of_updates = {'std': 0, 'decay': 0, 'margin': 0, 'average': 0}
    epoch = [i for i in range(0, 20)]
    for mode in modes:
        accuracy_per_epoch = list()
        highest_epoch = 0
        highest_accuracy = 0.0
        for i in range(0, 20):
            p = Perceptron()
            p.read_file('dataset/diabetes.train')
            (examples, labels) = data
            p.train(mode=mode, r_0=best_rate[mode], mu=best_mu[mode], t=i+1)
            number_of_updates[mode] = p.num_updates
            labels_predicted = p.predict(examples)
            correct = 0.0
            for j in range(0, len(labels)):
                if labels[j] == labels_predicted[j]:
                    correct += 1.0
            curr_acc = correct/float(len(labels_predicted))
            if curr_acc >= highest_accuracy:
                highest_accuracy = curr_acc
                highest_epoch = i
            accuracy_per_epoch.append(curr_acc)

        # plt.plot(epoch, accuracy_per_epoch)
        # plt.show()
        best_epoch[mode] = highest_epoch
        print("FOR MODE", mode, ":\nHIGHEST ACCURACY:", highest_accuracy, " on\nEPOCH:", highest_epoch,"\nNumber of updates:", number_of_updates[mode])
        print("#####################################################################################")

    for mode in modes:
        p = Perceptron()
        p.read_file('dataset/diabetes.train')
        p.train(mode=mode, mu=best_mu[mode], r_0=best_rate[mode], t=best_epoch[mode])
        data = read_file('dataset/diabetes.test')
        labels = data[1]
        examples = data[0]
        labels_predicted = p.predict(examples)
        correct = 0.0
        for i in range(0, len(labels_predicted)):
            if labels_predicted[i] == labels[i]:
                correct += 1.0
        curr_acc = correct/float(len(labels))
        print(f"Accuracy on diabetes.test for mode {mode}: {curr_acc*100.0:.2f}%\n")
        data = read_file('dataset/diabetes.dev')
        labels = data[1]
        examples = data[0]
        labels_predicted = p.predict(examples)
        correct = 0.0
        for i in range(0, len(labels_predicted)):
            if labels_predicted[i] == labels[i]:
                correct += 1.0
        curr_acc = correct / float(len(labels))
        print(f"Accuracy on diabetes.dev for mode {mode}: {curr_acc*100.0:.2f}%\n")

    dev = read_file('dataset/diabetes.dev')
    labelsd = dev[1]
    number_d = labelsd.count(-1)
    if float(number_d) / float(len(labelsd)) > 0.5:
        most_freq_dev = -1
    else:
        most_freq_dev = 1

    test = read_file('dataset/diabetes.test')
    labels = test[1]
    number = labels.count(-1)
    if float(number) / float(len(labels)) > 0.5:
        most_freq_test = -1
    else:
        most_freq_test = 1
    print("Accuracy for most frequent on dev= {:.2f}".format(float(labelsd.count(most_freq_dev)) / float(len(labelsd))))
    print("Accuracy for most frequent on test= {}".format(float(labels.count(most_freq_test)) / float(len(labels))))

    EPOCHS = 63

    for mode in modes:
        p_highest = Perceptron()
        p_highest.read_file('dataset/diabetes.train')
        p_highest.train(mode=mode, r_0=0.01, mu=0.01, t=EPOCHS)
        data = read_file('dataset/diabetes.test')
        x = data[0]
        labels = data[1]
        labels_predicted = p_highest.predict(x)
        correct = 0.0
        for i in range(0, len(labels_predicted)):
            if labels_predicted[i] == labels[i]:
                correct += 1.0
        curr_acc = correct / float(len(labels))
        print(f"Accuracy on diabetes.test for mode {mode} after {EPOCHS} epochs: {curr_acc*100.0:.2f}%")
