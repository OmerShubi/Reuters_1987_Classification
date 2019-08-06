import pickle

import numpy as np

import Calculations
import KNN
import parsing
import File_reader


class Model:
    def __init__(self, path_to_precooked_data):
        raw_data = parsing.parsing_data(path_to_precooked_data, False)
        print("finished parsing")
        data = File_reader.File_reader(raw_data)
        self.inv_labels = data.inv_labels
        self.train_features, self.train_labels = data.build_set_tfidf()
        # TODO remove bfore submission
        with open("train_features", "ab") as filename:
            pickle.dump(self.train_features, filename)
        with open("train_labels", "ab") as filename:
            pickle.dump(self.train_labels, filename)
        ### Till here

    def predict(self, path_to_test_set):
        predictions = []
        k = 5
        raw_test = parsing.parsing_data(path_to_test_set)
        data_test = File_reader.File_reader(raw_test, istest=True)
        test_features = data_test.build_set_tfidf()
        for index in range(test_features.shape[0]):
            instance = test_features[index]
            binary_predictions = self.KNN_predict(instance, k)
            labels = self.labels_from_prediction(binary_predictions)
            predictions.append(labels)
        # TODO Delete
        #  print(Calculations.f1_score())
        return tuple(predictions)

    def labels_from_prediction(self, binary_predictions):
        predicted_labels = []
        indexes = np.where(binary_predictions)[0]
        for index in indexes:
            predicted_labels.append(self.inv_labels[index])
        return tuple(predicted_labels)

    def KNN_predict(self):
        pass

    def k_nearest_neighbors(training_set, test_instance, training_labels, k):
        """
        kNN algorithm. Returns proposed label of a given test image 'test_instance', by finding the
        'k' similar neighbors (euclidean distance) for 'training_set' set of images.
        """
        closest_neighbors_labels = []
        distances = []

        for x in range(np.ma.size(training_set, 1)):
            dist = Calculations.cosine_similarity()euclidean_distance(training_set[:, x].tolist(), test_instance)
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
