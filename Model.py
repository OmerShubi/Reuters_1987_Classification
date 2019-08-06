import pickle

import numpy as np

import KNN
import parsing
import File_reader


class Model:
    def __init__(self, path_to_precooked_data):
        raw_data = parsing.parsing_data("/reuters_train_data")
        print("finished parsing")
        data = File_reader.File_reader(raw_data)
        self.inv_labels = data.inv_labels
        self.train_features, self.train_labels = data.build_set_tfidf()
        #TODO remove bfore submission
        with open('train_features', 'ab') as filename:
            pickle.dump(self.train_features, filename)
        with open('train_labels', 'ab') as filename:
            pickle.dump(self.train_labels, filename)
        ### Till here

    def predict(self, path_to_test_set):
        predictions = []
        k=5
        raw_test = parsing.parsing_data(path_to_test_set)
        data_test = File_reader.File_reader(raw_test, istest=True)
        test_features = data_test.build_set_tfidf()
        for index in range(test_features.shape[0]):
            instance = test_features[index]
            binary_predictions = KNN.predict(self.train_features, self.train_labels, instance, k)
            labels = self.labels_from_prediction(binary_predictions)
            predictions.append(labels)
        return tuple(predictions)


    def labels_from_prediction(self, binary_predictions):
        predicted_labels = []
        indexes = np.where(binary_predictions)[0]
        for index in indexes:
            predicted_labels.append(self.inv_labels[index])
        return tuple(predicted_labels)

