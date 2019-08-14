import logging.config

import model


def main():
    """

    :return:
    """

    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    logger.info("********** NEW RUN **********")

    # *******Change debug to True for small dataset********
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
    predictions = knn_model.predict(test_data_dir)

    logger.info("Prediction complete")

    print(predictions)


if __name__ == "__main__":
    main()
