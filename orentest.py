import re
city_country_list = []
with open("cities.csv") as f:
    for line in f:
        city_country_list.append(line.split())

labels_pool = ['china']
city_labels = []
current_labels = []
for list in city_country_list:
    if list[1] in labels_pool:
        city_labels.append(list[0])
        current_labels.append((list[1]))







print(city_labels)