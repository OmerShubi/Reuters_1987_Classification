import re
import math
import numpy as np


class File_reader:
    def __init__(self, data_articles, istest=False):
        self.data_articles = data_articles
        self.istest = istest
        self.number_of_docs = len(self.data_articles)
        self.df = {}
        self.words = {}
        self.stop_words = []
        self.labels = {}
        self.create_stop_words_list()
        self.create_words_bank()
        self.inv_labels = {v: k for k, v in self.labels.items()}

    def create_stop_words_list(self):
        with open("./stop_words.txt") as stop_words_file:
            for stop_word in stop_words_file:
                self.stop_words.append(stop_word.rstrip())

    # TODO: lemmatization?
    def pre_process(self, word):
        """
		Preprocesses a word by turning it to lower case,
		removes some punctuation and returns the word or
		empty string if it is a stop word
		:param word: string
		:return: updated word - string
		"""
        strip_specail_chars = re.compile("[^A-Za-z ]+")
        word = word.lower().replace("<br />", " ")
        word = re.sub(strip_specail_chars, "", word.lower())
        # returns empty string if word is a stop word
        if word in self.stop_words:
            return ""
        return word

    def create_words_bank(self):
        """
		create_words_bank from all the articles and pull of labels
		:return:
		"""
        index = 0
        index2 = 0
        for article in self.data_articles:
            seen_in_this_article = []
            for word in article["text"].split():
                word = self.pre_process(word)
                if word == "":
                    continue
                if word not in self.df:
                    self.df[word] = 1  # document frequency
                    seen_in_this_article.append(word)
                if word not in seen_in_this_article:
                    self.df[word] += 1
                    seen_in_this_article.append(word)
                if (
                        word not in self.words.keys()
                ):  # if the word doesnt already exists in the words dictionary
                    self.words[word] = index  # add it
                    index += 1
            # create pull labels
            for label in article["labels"]:
                if label not in self.labels.keys():  # if the label doesnt already exists in the labels dictionary
                    self.labels[label] = index2  # add it
                    index2 += 1
        # print(self.labels)

    def build_set_tfidf(self):
        """
        Builds the data vector using tfidf format
		:return: the file in vector form, using tfidf format
        """
        doc_set = []
        labels_set = []
        for article in self.data_articles:
            vec = len(self.words) * [0.0]
            for word in article["text"].split():
                word = self.pre_process(word)
                if word == "":
                    continue
                vec[self.words[word]] += 1
            # After iterating over all words we now have the tf and can store words in tfidf format
            for word in self.words.keys():
                if vec[self.words[word]] == 0:
                    continue
                else:
                    vec[self.words[word]] = vec[self.words[word]] * math.log(
                        (self.number_of_docs / self.df[word]), 10)
            doc_set.append(vec)
            vec_labels = len(self.labels) * [0]
            for label in article["labels"]:
                vec_labels[self.labels[label]] = 1
            labels_set.append(vec_labels)
        return np.array(doc_set), np.array(labels_set)

    def parse_test(self, test_articles):
        """
        Builds the data vector using tfidf format
        :param file_to_vector: the file to be processed
        :return: the file in vector form, using tfidf format
        """
        doc_set = []
        for article in test_articles:
            vec = len(self.words) * [0, ]
            for word in article['text'].split():
                word = self.pre_process(word)
                if word == '':
                    continue
                if word in self.words:
                    vec[self.words[word]] += 1
            # After iterating over all words we now have the tf and can store words in tfidf format
            for word in self.words.keys():
                if vec[self.words[word]] == 0:
                    continue
                else:
                    vec[self.words[word]] = vec[self.words[word]] * math.log((self.number_of_docs / self.df[word]), 10)
            doc_set.append(vec)
        return np.array(doc_set)
