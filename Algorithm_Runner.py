import math
import numpy as np
import KNN
import NearestCentroid

NEIGHBORS = 10


class Algorithm_Runner:
    def __init__(self, classifier):
        """
        Initializes the AlgorithRunner with the desired classifier
        :param classifier_method: desired classifier, expects 'KNN' or 'Rocchio'
        """
        self.algorithm = self.select_model(classifier)
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    @staticmethod
    def select_model(classifier_method):
        """
        Initializes desired classifier
        :param classifier_method: desired classifier, expects 'KNN' or 'Rocchio'
        :return: classifier sklearn object
        """
        if classifier_method == "KNN":
            return KNN(n_neighbors=NEIGHBORS)
        elif classifier_method == "Rocchio":
            return NearestCentroid()

    def fit(self, x_train, y_train):
        """
        Fit the  model according to the given training data.
        :param x_train: training data
        :param y_train: training labels
        :return: the the specific algorithm fit
        """
        self.algorithm.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Perform classification on an array of test vectors x_test
        :param x_test: test data
        :return: The predicted class C for each sample in x_test
        """
        return self.algorithm.predict(x_test)

    def cross_val(self):
        pass

    def calc_accuracy(test_set, classifier, distance_method):
        """
        Calculates a given classifier's accuracy on a test set, using the chosen distance method (euclidean / cosine)
        :param test_set: the test set, dictionary
        :param classifier: the trained rocchio classifier
        :param distance_method:
        :return:
        """
        correct = 0.0
        total = len(test_set.keys())
        for key in test_set:
            real = test_set[key][-1]
            predicted = classifier.predict(test_set[key][0:-1], distance_method)
            if real == predicted:
                correct += 1.0
        return correct / total

    @staticmethod
    def f1_score(expected, predicted):
        tp = 0
        for e, p in zip(expected, predicted):
            if e == p and p == 1:
                tp += 1
        tp_plus_fp = np.sum(predicted)
        tp_plus_fn = np.sum(expected)

        recall = tp/tp_plus_fn
        precision = tp/tp_plus_fp
        print("recall:", recall)
        print("precision:", precision)
        f1 = 2*precision*recall/(recall+precision)
        return f1
