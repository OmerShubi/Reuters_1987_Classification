@staticmethod
def create_cities_dict(labels_pool):
    """

    :param labels_pool:
    :return: cities to country dictionary
    """
    city_country_list = []
    with open("world-cities_csv.csv", encoding="iso-8859-1") as f:
        for line in f:
            city_country_list.append(list(map(lambda x: x.replace('\n', ""), line.split(','))))

    current_labels = []
    for city_country in city_country_list:
        if city_country[1] in labels_pool:
            clean_list = list(map(lambda x: x.replace('\n', ""), city_country))
            current_labels.append(clean_list)

    cities_dict = {elem[0]:elem[1] for elem in current_labels}

    return cities_dict


    citylabel= self.train_features[index]["DATELINE"]
    dict = create_cities_dict
    if citylabel in dict.keys():
        if dict[citylabel] not in predicted_labels:
            predicted_labels.append(dict[citylabel])

print(create_cities_dict(["china","usa"]))
# TODO check about r0009.sgm
# TODO Dateline
# TODO pickle
# TODO model.model
#TODO delete prints
