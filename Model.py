import pickle

import KNN
import parsing
import File_reader


class Model:
    def __init__(self, path_to_precooked_data):
        raw_data = parsing.parsing_data("/reuters_train_data")
        print("finished parsing")
        data = File_reader.File_reader(raw_data)
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
            prediction_binary = KNN.predict(self.train_features, self.train_labels, instance, k)
            labels = self.labels_from_prediction(prediction_binary)
            predictions.append(labels)
        return predictions
