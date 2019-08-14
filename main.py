import model

TRAIN_DATA_DIR = "reuters_train_data"
TEST_DATA_DIR = "reuters_test_data"


def main():
    """

    :return:
    """
    knn_model = model.Model(TRAIN_DATA_DIR)
    predictions = knn_model.predict(TEST_DATA_DIR)
    print(predictions)


if __name__ == "__main__":
    main()
