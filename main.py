import parsing
import File_reader
import Model

def main():

    model = Model('.')
    predictions = model.predict("reuters_test_data")

    rawdata = parsing.parsing_data("Raw Data - DO NOT CHANGE/reuters_train_data")
    print("finished parsing")
    data = File_reader.File_reader(rawdata)
    train_features_raw, train_labels_raw = data.build_set_tfidf()





if __name__ == "__main__":
    main()
