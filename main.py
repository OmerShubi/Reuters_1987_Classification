import Model
import pickle
import numpy as np
def main():
    # model = Model.Model("reuters_train_data")
    # predictions = model.predict("reuters_test_data")
    # print(predictions)
    with open('train_labels', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    print(np.array(data))

if __name__ == "__main__":
    main()
