"""
---Parsing Tool---
This script allows the user to parse a Reuters XML file.
This script requires that 'xmljson' be installed within the Python
environment you are running this script in.
"""
import logging
import os
from xml.etree.ElementTree import fromstring

from xmljson import parker as pr

logger = logging.getLogger(__name__)


class DataParser:
    def __init__(self, directory_path, is_test_data=False):
        self.data_path = directory_path
        self.is_test_data = is_test_data

    def parse_data(self, is_test=False):
        """
        :param is_test: boolean. Gets True if the article is part of the test set.
        :return: Returns list of dictionaries with TEXT and LABELS keys for each article
        """
        final_data = []

        for root, dirs, files in os.walk(self.data_path, topdown=False):
            for name in files:
                try:
                    data = list(filter(lambda x: x['labels'] != [] and x['text'] != '',
                                       DataParser._parsing(os.path.join(root, name), is_test)))

                    logger.debug("number of articles in file {}: {}".format(name, len(data)))

                    final_data = final_data + data
                except UnicodeDecodeError:
                    continue

        logger.info("Total number of articles parsed: %s", len(final_data))

        return final_data

    @staticmethod
    def _create_labels(article):
        """
        :param article:(ordered dictionary).
        :return:list of labels based on existing labels in the received article.
        """
        labels = []
        labels_type = ["TOPICS", "PLACES", "PEOPLE", "ORGS", "EXCHANGES", "COMPANIES"]

        for label_type in labels_type:
            if article[label_type] is not None:
                if article[label_type]['D'] != "" and article[label_type]['D'] is not None:
                    labels.append(article[label_type]['D'])

        # Flatting the label list in case labels is list of lists
        flat_labels = []
        for sublist in labels:
            if isinstance(sublist, list):
                for item in sublist:
                    flat_labels.append(item)
            else:
                flat_labels.append(sublist)
        return flat_labels

    @staticmethod
    def _get_dateline(article):
        """
        :param article:
        :return:
        """
        if 'DATELINE' in article['TEXT']:
            return article['TEXT']['DATELINE'].split(',')[0]
        else:
            return ""

    @staticmethod
    def _get_text(article, is_test=False):
        """
        :param article:(ordered dictionary).
        :param is_test: if test data parses without the 'title' attribute
        :return:Returns the text of the received article.
        """
        text_to_return = ""
        if not is_test:
            if 'TITLE' in article['TEXT']:
                text_to_return = text_to_return + " " + article['TEXT']['TITLE']
        if 'BODY' in article['TEXT']:
            text_to_return = text_to_return + " " + article['TEXT']['BODY']
        return text_to_return

    @staticmethod
    def _parsing(file_path, test):
        """
        :param file_path:
        :return:
        """
        with open(file_path) as f:
            f.readline()
            raw_data = "<xml>" + f.read() + "</xml>"
            raw_data = raw_data.replace('&', "").replace('#', "")
            data_dict = pr.data(fromstring(raw_data), preserve_root=True)
            if test:
                data = [
                    {"labels": "", "text": DataParser._get_text(article, test), "dateline":
                        DataParser._get_dateline(article)} for article in data_dict['xml']['REUTERS']]
            else:
                data = [{"labels": DataParser._create_labels(article), "text": DataParser._get_text(article, test),
                         "dateline": DataParser._get_dateline(article)} for article in data_dict['xml']['REUTERS']]
            return data

    # print(parsing_data('test', False)[0])
    # print(parsing_data('test', False)[223])
    # for elem in parsing_data('test', False):
    #     print(elem)
