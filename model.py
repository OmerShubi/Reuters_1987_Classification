import logging
import os

import numpy as np

import FileReader as File_reader
import dataParser

logger = logging.getLogger(__name__)

KNN_NEIGHBORS = 3
WORLD_CITIES_PATH = "Data/world-cities.csv"


class Model:

    def __init__(self, path_train_dir, is_submission=False):

        if is_submission:
            path = os.path.join('.', "reuters_train_data")
        else:
            path = path_train_dir

        logger.info('Parsing train data...')
        data_parser = dataParser.DataParser(path)
        raw_data = data_parser.parse_data()

        logger.info('parse train data COMPLETE')

        # process data
        self.data = File_reader.FileReader(raw_data)
        # self.data = pickle.load(open("data.zip", 'rb'))

        self.inv_labels = self.data.inv_labels
        # self.inv_labels = pickle.load(open("inv_labels.zip", 'rb'))

        logger.info('Creating train_features and train_labels...')

        self.train_features, self.train_labels = self.data.build_set_tfidf()
        # self.train_features = self.data.build_set_tfidf()
        # self.train_labels = pickle.load(open("train_labels.p", 'rb'))

        logger.info('Creating train_features and train_labels COMPLETE')
        logger.info('Number of train articles: %s', self.train_features.shape[0])

        # TODO unzip pickles, comment above and uncomment below
        # Restoring train features and labels from pickle..
        # self.train_features = pickle.load(open("train_features.p", 'rb'))
        # self.train_labels = pickle.load(open("train_labels.p", 'rb'))

    def predict(self, path_to_test_set):
        """
        For each article in each file in path_to_test_set (dir) predicts the labels of the article
        :param path_to_test_set: directory with all the test reuters files
        :return: tuple of tuples, each inner tuple stores the labels of an article.
                    Outer tuple is ordered, inner is not
        """
        predictions = []

        k = KNN_NEIGHBORS

        logger.info('Parsing test data...')

        data_parser = dataParser.DataParser(path_to_test_set)
        raw_test = data_parser.parse_data(is_test=True)

        logger.info('parse test data COMPLETE')

        logger.info('Creating test_features...')

        test_features = self.data.parse_test(raw_test)

        logger.info('Running KNN with k = %s ...', k)

        cities_countries = Model.create_cities_dict(list(self.data.labels.keys()))
        for index in range(test_features.shape[0]):
            instance = test_features[index]
            binary_predictions = self.knn_predict(instance, k)
            labels = self.labels_from_prediction(binary_predictions)
            city_label = raw_test[index]["dateline"].replace(" ", "")

            if city_label in cities_countries.keys():
                if cities_countries[city_label] not in labels:
                    labels.append(cities_countries[city_label])
            predictions.append(tuple(labels))

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
        return predicted_labels

    @staticmethod
    def create_cities_dict(labels_pool):
        """
        :param labels_pool:
        :return: cities to country dictionary
        """
        city_country_list = []
        with open(WORLD_CITIES_PATH, encoding="iso-8859-1") as f:
            for line in f:
                city_country_list.append(list(map(lambda x: x.replace('\n', ""), line.split(','))))

        current_labels = []
        for city_country in city_country_list:
            if city_country[1] in labels_pool:
                clean_list = list(map(lambda x: x.replace('\n', ""), city_country))
                current_labels.append(clean_list)

        cities_dict = {elem[0]: elem[1] for elem in current_labels}

        return cities_dict

    def knn_predict(self, instance, k):
        """

        :param instance:
        :param k:
        :return:
        """
        closest_neighbors_labels = self.k_nearest_neighbors(instance, k)
        return Model.best_neighbor_match_check(closest_neighbors_labels, k)

    def k_nearest_neighbors(self, test_instance, k):
        """
        kNN algorithm. Returns proposed label of a given test image 'test_instance', by finding the
        'k' similar neighbors (euclidean distance) for 'training_set' set of images.
        """
        closest_neighbors_labels = []
        distances = []

        length = np.ma.size(self.train_features, 0) - 1
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
        length = k_neighbors_labels.shape[1] - 1
        labels = []
        for index in range(length):
            k_neighbors_label = k_neighbors_labels[:, index]
            if (k_neighbors_label.sum() / k) > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    @staticmethod
    def cosine_distance(x, y):

        """
        Calculates cosine similarity between two lists

        Assumes lists are of same length
        :param x: list
        :param y: list
        :return: cosine similarity
        """

        base = (np.linalg.norm(x) * np.linalg.norm(y))
        if base != 0:
            return 1 - (np.dot(x, y) / base)
        return 1
