import pickle
import numpy as np
import PShSh_Submission.File_reader as File_reader
import PShSh_Submission.parsing as parsing
NEIGHBORS = 5
import os


class Model:
    def __init__(self, path_train_dir):

        path = os.path.join(path_train_dir,"train_data")
        # Parsing train data...
        raw_data = parsing.parsing_data(path, False)

        # parse train data COMPLETE
        self.data = File_reader.File_reader(raw_data)
        # self.data = pickle.load(open("data.zip", 'rb'))

        self.inv_labels = self.data.inv_labels
        # self.inv_labels = pickle.load(open("inv_labels.zip", 'rb'))

        # Creating train_features and train_labels...
        self.train_features, self.train_labels = self.data.build_set_tfidf()
        # self.train_features = self.data.build_set_tfidf()
        # self.train_labels = pickle.load(open("train_labels.p", 'rb'))

        # Creating train_features and train_labels COMPLETE
        # print(Number of train articles:", self.train_features.shape[0])

        # TODO unzip pickles, comment above and uncomment below
        # Restoring train features and labels from pickle..
        # self.train_features = pickle.load(open("train_features.p", 'rb'))
        # self.train_labels = pickle.load(open("train_labels.p", 'rb'))

    def predict(self, path_to_test_set):
        """

        :param path_to_test_set:
        :return:
        """
        predictions = []
        k = NEIGHBORS
        # Parsing test data...
        raw_test = parsing.parsing_data(path_to_test_set, True)

        # parse test data COMPLETE
        # Creating test_features...
        test_features = self.data.parse_test(raw_test)

        # Creating test_features COMPLETE
        # Running KNN...
        for index in range(test_features.shape[0]):
            instance = test_features[index]
            binary_predictions = self.knn_predict(instance, k)
            labels = self.labels_from_prediction(binary_predictions)
            predictions.append(labels)

        return tuple(predictions)

    def labels_from_prediction(self, binary_predictions):
        """

        :param binary_predictions:
        :return:
        """
        predicted_labels = []
        indexes = np.where(binary_predictions)[0]
        for index in indexes:
            predicted_labels.append(self.inv_labels[index])
        return tuple(predicted_labels)

    def knn_predict(self, instance, k):
        """

        :param instance:
        :param k:
        :return:
        """
        closest_neighbors_labels = self.k_nearest_neighbors(instance, k)
        return Model.best_neighbor_match_check(closest_neighbors_labels,k)

    def k_nearest_neighbors(self, test_instance, k):
        """
        kNN algorithm. Returns proposed label of a given test image 'test_instance', by finding the
        'k' similar neighbors (euclidean distance) for 'training_set' set of images.
        """
        closest_neighbors_labels = []
        distances = []

        length = np.ma.size(self.train_features, 0)-1
        for i in range(length):
            dist = Model.cosine_distance(self.train_features[i], test_instance)
            distances.append(dist)
        max_dist = max(distances)
        distances = np.array(distances, dtype=float)
        for neighbor in range(k):
            closest_neighbor = np.argmin(distances)
            closest_neighbors_labels.append(self.train_labels[closest_neighbor, :])
            distances[closest_neighbor] = max_dist

        return np.array(closest_neighbors_labels)

    @staticmethod
    def best_neighbor_match_check(k_neighbors_labels, k):
        """

        :param k_neighbors_labels:
        :param k:
        :return:
        """
        """	Returns the values with the most repetitions in `k_neighbors`. """
        length = k_neighbors_labels.shape[1]-1
        labels = []
        for index in range(length):
            k_neighbors_label = k_neighbors_labels[:, index]
            if (k_neighbors_label.sum()/k)>0.5:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    @staticmethod
    def cosine_distance(list1, list2):
        """
        Calculates cosine similarity between two lists

        Assumes lists are of same length
        :param list1: list
        :param list2: list
        :return: cosine similarity
        """
        return 1-(np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2)))
