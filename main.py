import model

def main():
    model1 = model.Model('.')
    predictions = model1.predict("reuters_test_data")
    print(predictions)


if __name__ == "__main__":
    main()
