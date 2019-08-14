import model
import logging
import logging.config


def main():
    """

    :return:
    """

    # Gets or creates a logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger(__name__)

    logger.info("********** NEW RUN **********")

    # Change debug to true for small dataset
    debug = True

    if debug:
        train_data_dir = "train_data"
        test_data_dir = "reuters_test_data"
    else:
        train_data_dir = "reuters_train_data"
        test_data_dir = "reuters_test_data"

    logger.info("Initiating training with data from '%s' directory", train_data_dir)
    knn_model = model.Model(train_data_dir)

    logger.info("Predicting testing with data from '%s' directory", test_data_dir)
    predictions = knn_model.predict(test_data_dir)

    print(predictions)


if __name__ == "__main__":
    main()
