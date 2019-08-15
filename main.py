import pickle
import logging.config

import sklearn.metrics
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import date

import model


def main():
    """

    :return:
    """

    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    logger.info("********** NEW RUN **********")

    # *******Change debug to True for small dataset ********
    debug = False

    if debug:
        train_data_dir = "Data/train_data"
        test_data_dir = "Data/reuters_test_data"
    else:
        train_data_dir = "Data/reuters_train_data"
        test_data_dir = "Data/reuters_test_data"

    logger.info("Initiating training with data from '%s' directory", train_data_dir)
    knn_model = model.Model(train_data_dir)

    logger.info("Predicting testing with data from '%s' directory", test_data_dir)
    # predictions = knn_model.predict(test_data_dir) TODO
    predictions, reference = knn_model.predict(test_data_dir)

    logger.info("Prediction complete")

    name = "Pickles/"+get_file_name_datetime("predictions")
    pickle.dump(predictions, name)
    logger.info("Prediction saved to %s", name)

    print(predictions)
    print(reference)
    mlb = MultiLabelBinarizer()
    r = mlb.fit_transform(reference)
    p = mlb.transform(predictions)
    try:
        score = sklearn.metrics.f1_score(y_true=r, y_pred=p, average='macro')
        print(score)
        logger.info("The f1 score is: %s", score)
    except ValueError as ex:
        logger.error("result value is invalid: " + str(ex))


if __name__ == "__main__":
    main()


def get_file_name_datetime(prefix, suffix='', file_type='p'):
    return prefix + "-" + str(date.today()) + suffix + "." + file_type
