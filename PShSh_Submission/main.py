import PShSh_Submission.model

def main():
    model1 = PShSh_Submission.model.Model("train_data")
    predictions = model1.predict("reuters_test_data")
    print(predictions)

if __name__ == "__main__":
    main()
