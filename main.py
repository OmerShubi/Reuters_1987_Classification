import Model
import pickle
import numpy as np
def main():
    model = Model.Model("reuters_train_data")
    predictions = model.predict("reuters_test_data")
    print(predictions)
    model = Model.Model("train")
    predictions = model.predict_f1("test_not_oren")
    print(predictions)


if __name__ == "__main__":
    main()
