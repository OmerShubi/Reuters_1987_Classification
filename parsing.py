"""---Parsing Tool---
This script allows the user to parse a Reuters XML file.

This script requires that 'xmljson' be installed within the Python
environment you are running this script in.
"""
from xmljson import parker as pr
from xml.etree.ElementTree import fromstring
import json
import types
import os


def create_labels(article):
    """ Gets an article object (ordered dictionary).
        Returns a list of labels based on existing labels in the received article. """

    labels = []
    labels_type = ["TOPICS", "PLACES", "PEOPLE", "ORGS", "EXCHANGES", "COMPANIES"]

    for label_type in labels_type:
        if article[label_type] is not None:
            labels.append(article[label_type]['D'])

    # Flats the label list in case labels is list of lists
    flat_labels = []
    for sublist in labels:
        if isinstance(sublist, list):
            for item in sublist:
                flat_labels.append(item)
        else: flat_labels.append(sublist)
    return flat_labels


def get_text(article):
    """ Gets an article object (ordered dictionary).
        Returns the text of the received article. """

    text_to_return = ""
    if 'TITLE' in article['TEXT']:
        text_to_return = text_to_return + article['TEXT']['TITLE']
    if 'BODY' in article['TEXT']:
        text_to_return = text_to_return + article['TEXT']['BODY']
    return text_to_return


def parsing(file_path):
    with open(file_path) as f:
        f.readline()
        raw_data = "<xml>" + f.read() + "</xml>"
        raw_data = raw_data.replace('&', "").replace('#', "")
        data_dict = pr.data(fromstring(raw_data), preserve_root=True)
        data = [{"labels": create_labels(article), "text": get_text(article)} for article in data_dict['xml']['REUTERS'] ]
        return data


final_data = []
first_file = True
for root, dirs, files in os.walk("Raw Data - DO NOT CHANGE/reuters_train_data", topdown=False):
    for name in files:
        if first_file:
            first_file = False
            continue
        final_data = final_data + parsing(os.path.join(root, name))

