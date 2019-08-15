import logging.config
import sklearn.metrics
from sklearn.preprocessing import MultiLabelBinarizer

import model
import pickleHelper

def main():
    """

    :return:
    """

    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    logger.info("********** NEW RUN **********")

    # *******Change debug to True for small dataset ********
    debug = True

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

    pickleHelper.save_to_pickle("predictions", predictions)
    # path_to_predictions = "Pickles/predictions-2019-08-15-1027.p"
    # try:
    #     returned_predictions = pickleHelper.retrieve_from_pickle(path_to_predictions, "predictions")
    # except FileNotFoundError:
    #     returned_predictions = knn_model.predict(test_data_dir)

    print(predictions)
    # print(returned_predictions)
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


