import parsing
import File_reader

def main():
    rawdata = parsing.parsing_data("Raw Data - DO NOT CHANGE/reuters_train_data")
    print("finished parsing")
    data = File_reader.File_reader(rawdata)
    data.build_set_tfidf()

if __name__ == "__main__":

    main()

