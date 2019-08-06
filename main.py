import numpy as np

import parsing
import File_reader
import Model

def main():

    model = Model.Model("reuters_train_data")
    predictions = model.predict("reuters_test_data")
    print(predictions)


if __name__ == "__main__":
    main()
