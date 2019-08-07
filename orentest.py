def create_cities_dict(labels_pool):
    city_country_list = []
    with open("cities.csv") as f:
        for line in f:
            city_country_list.append(line.split())


    current_labels = []
    for city_country in city_country_list:
        if city_country[1] in labels_pool:
            clean_list = list(map(lambda x: x.replace(',', ""), city_country))
            current_labels.append(clean_list)

    cities_dict = {elem[0]:elem[1] for elem in current_labels}

    return cities_dict
print(create_cities_dict(["china","usa"]))
# TODO check about r0009.sgm
# TODO Dateline
# TODO pickle
# TODO model.model
#TODO delete prints

