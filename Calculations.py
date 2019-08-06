import numpy as np

def cosine_similarity(list1, list2):
    """
    Calculates cosine similarity between two lists

    Assumes lists are of same length
    :param list1: list
    :param list2: list
    :return: cosine similarity
    """
    return 1-(np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2)))


def distances(test, train):
    """
    Calculates distances between test and train

    :param test:
    :param train:
    :return: distances list
    """
    distances = []
    for i in range(len(test)):
        distances.append(cosine_similarity(test[i], train))
    return distances



