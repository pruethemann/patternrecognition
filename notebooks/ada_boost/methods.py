# this file is implementing a method classify with an ada boost

from Ada_boost import AdaBoost
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import time


def test_classification(ada_boost: AdaBoost, data_test):
    m, d = np.shape(data_test)
    fp = 0
    fn = 0
    corr = 0
    for i in range(m):
        if ada_boost.classify(data_test[i, :d-1]) == data_test[i, d - 1]:
            corr += 1
        elif ada_boost.classify(data_test[i, :d-1]) < data_test[i, d-1]:
            fn += 1
        else:
            fp += 1

    print('Ada-Boost Classified ', corr, 'out of ', m, 'examples correctly.')
    print('False negative: ', fn)
    print('False positive: ', fp)
    print()

	
def load_data():
    toy_train = np.transpose(np.load('inputs.npy'))
    toy_train_labels = np.load('targets.npy')
    toy_train_labels[toy_train_labels == 0] = -1
    toy_train = np.append(toy_train, np.reshape(toy_train_labels, (np.shape(toy_train)[0], 1)), axis=1)
    n, d = np.shape(toy_train)

    toy_test = np.transpose(np.load('validation_inputs.npy'))
    toy_test_labels = np.load('validation_targets.npy')
    toy_test_labels[toy_test_labels == 0] = -1
    toy_test = np.append(toy_test, np.reshape(toy_test_labels, (np.shape(toy_test)[0], 1)), axis=1)
    m, d = np.shape(toy_test)

    ind_1 = np.where(toy_train[:, 2] > 0)[0]
    data_1 = np.take(toy_train, ind_1, axis=0)

    ind_2 = np.where(toy_train[:, -1] < 0)[0]
    data_2 = np.take(toy_train, ind_2, axis=0)

    # simplify the situation a bit
    data_1[:, 1] += 0.2
    data_1[:, 0] += 0
    # corresponds to data_1: just move everything a bit
    ind_test_1 = np.where(toy_test[:, 2] > 0)[0]
    toy_test[ind_test_1, 1] += 0.2
    toy_test[ind_test_1, 0] += 0

    return data_1, data_2, toy_test


def formatize_weather_data(data_prime):
    data = data_prime.copy()

    # remove the features
    data = data[1:, :]

    # convert the prediction to a numerical number
    n, d = np.shape(data)
    ind_pos = []
    ind_neg = []
    for i in range(n):
        if data[i, -1] == 'Yes':
            data[i, -1] = 1
            ind_pos.append(i)
        elif data[i, -1] == 'No':
            data[i, -1] = -1
            ind_neg.append(i)

    # take numerical values
    data = data.astype(float)

    # split into two categories
    data_1 = np.take(data, ind_pos, axis=0)
    data_2 = np.take(data, ind_neg, axis=0)

    return data_1, data_2

