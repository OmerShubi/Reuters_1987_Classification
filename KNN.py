import numpy as np
import Calculations
import parsing


class KNN:
    def fit(self, features_train, labels_train):
        """
        Fit the  model according to the given training data.
        :param x_train: training data
        :param y_train: training labels
        :return: the the specific algorithm fit
        """
        self.features_train = np.ndarray(features_train)
        self.labels_train = np.ndarray(labels_train)

    def predict(self, test_data):
        """
        Perform classification on an array of test vectors x_test
        :param x_test: test data
        :return: The predicted class C for each sample in x_test
        """






    def k_nearest_neighbors(training_set, test_instance, training_labels, k):
        """
        kNN algorithm. Returns proposed label of a given test image 'test_instance', by finding the
        'k' similar neighbors (euclidean distance) for 'training_set' set of images.
        """
        closest_neighbors_labels = []
        distances = []

        for x in range(np.ma.size(training_set, 1)):
            dist = euclidean_distance(training_set[:, x].tolist(), test_instance)
            distances.append(dist)

        distances = np.array(distances, dtype=float)

        for neighbor in range(k):
            closest_neighbor = np.argmin(distances)
            closest_neighbors_labels.append(training_labels[closest_neighbor])
            distances[closest_neighbor] = distances.max()

        return closest_neighbors_labels


    def best_neighbor_match_check(k_neighbors_labels):
        """	Returns the values with the most repetitions in `k_neighbors`. """
        k_neighbors_labels.sort()
        longest_repeats = current_repeats = 0
        current_value = best_match_value = k_neighbors_labels[0]
        for value in k_neighbors_labels:
            if value == current_value:
                current_repeats += 1
            else:
                current_repeats = 1
                current_value = value
            if longest_repeats < current_repeats:
                longest_repeats = current_repeats
                best_match_value = current_value

        return best_match_value
