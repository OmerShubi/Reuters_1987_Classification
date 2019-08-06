import pickle

import numpy as np

import Calculations
import File_reader
import parsing


class Model:
    def __init__(self, path_to_precooked_data):
        print("Parsing train data...")
        raw_data = parsing.parsing_data(path_to_precooked_data, False)
        print("parse train data COMPLETE")
        data = File_reader.File_reader(raw_data)
        self.inv_labels = data.inv_labels
        print("Building tfidf set...")
        self.train_features, self.train_labels = data.build_set_tfidf()
        print("Build tfidf set COMPLETE")
        # TODO remove bfore submission
        with open("train_features", "ab") as filename:
            pickle.dump(self.train_features, filename)
        with open("train_labels", "ab") as filename:
            pickle.dump(self.train_labels, filename)
        ### Till here

    def predict(self, path_to_test_set):
        predictions = []
        k = 5
        raw_test = parsing.parsing_data(path_to_test_set, True)
        data_test = File_reader.File_reader(raw_test, istest=True)
        test_features = data_test.build_set_tfidf()
        for index in range(test_features.shape[0]):
            instance = test_features[index]
            binary_predictions = self.knn_predict(instance, k)
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

    def knn_predict(self, instance, k):
        closest_neighbors_labels = self.k_nearest_neighbors(instance, k)
        return self.best_neighbor_match_check(closest_neighbors_labels)

    def k_nearest_neighbors(self, test_instance, k):
        """
        kNN algorithm. Returns proposed label of a given test image 'test_instance', by finding the
        'k' similar neighbors (euclidean distance) for 'training_set' set of images.
        """
        closest_neighbors_labels = np.array
        distances = np.array(dtype=float)

        for i in range(np.ma.size(self.train_features, 1)):
            dist = Calculations.cosine_distance(
                self.train_features[i, :], test_instance
            )
            np.append(distances, dist)
        max_dist = distances.max()
        for neighbor in range(k):
            closest_neighbor = np.argmin(distances)
            np.append(closest_neighbors_labels, self.train_labels[closest_neighbor, :])
            distances[closest_neighbor] = max_dist

        return closest_neighbors_labels

    @staticmethod
    def best_neighbor_match_check(k_neighbors_labels):
        """	Returns the values with the most repetitions in `k_neighbors`. """
        length = k_neighbors_labels.shape[0]
        for index in range(length):
            k_neighbors_label = k_neighbors_labels[index, :].sort()
            longest_repeats = current_repeats = 0
            current_value = best_match_value = k_neighbors_label[0]
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
