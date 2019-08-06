import numpy as np

def cosine_distance(list1, list2):
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
        distances.append(cosine_distance(test[i], train))
    return distances


def f1_score_label(expected, predicted):
    """

    :param expected:  list of predictions for a specific label
    :param predicted:
    :return:
    """
    tp = 0
    for e, p in zip(expected, predicted):
        if e == p and p == 1:
            tp += 1
    tp_plus_fp = np.sum(predicted)
    tp_plus_fn = np.sum(expected)

    recall = tp/tp_plus_fn
    precision = tp/tp_plus_fp

    f1 = 2*precision*recall/(recall+precision)
    return f1


def f1_score(all_expected, all_predicted):
    """

    :param all_expected:  expects list of list
    :param all_predicted:
    :return:
    """
    f1 = 0
    for expected, predicted in zip(all_expected, all_predicted):
        f1 += f1_score_label(expected,predicted)
    return f1/len(all_predicted)

