import sys
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from typing import List
from logreg import LOGREG

matplotlib.use('TkAgg')

dataPath = '../data/'


def get_rbg_image(image: np.ndarray, width: int = 32) -> np.ndarray:
    img = np.zeros((width, width, 3))
    img[:, :, 0] = np.reshape(image[:1024], (width, width))
    img[:, :, 1] = np.reshape(image[1024:2048], (width, width))
    img[:, :, 2] = np.reshape(image[2048:], (width, width))
    return img


def figurePlotting(imgarray: np.ndarray, N: int, is_cifar: bool = False, name='', random=True) -> None:
    '''
    CIFAR image visualization - rescaling the vector images to 32x32 and visualizes in a matplotlib plot
    :param imgarray: Array of images to be visualized, each column is an image
    :param N: Number of images per row/column
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    :param name: Optional name of the plot
    :param random: True if the images should be taken randomly from the array - otherwise start of the array is taken
    '''
    plt.figure(name)
    for i in range(0, N * N):
        imgIndex = i
        if random:
            imgIndex = np.random.randint(low=0, high=imgarray.shape[1])
        if is_cifar:
            img = get_rbg_image(imgarray[:, imgIndex])
            plt.subplot(N, N, i + 1)
            plt.imshow(img)
            plt.axis('off')
        else:
            img = np.reshape(imgarray[:, imgIndex], (16, 16))
            plt.subplot(N, N, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')


def visualizeClassification(data: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
                            num: int, is_cifar: bool = False, name: str = '') -> None:
    '''
    Use LOGREG classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: CIFAR data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of CIFAR images to show
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    :param name: Optional name of the plot
    '''
    res = np.abs(predictions - labels)
    number_of_misses = int(np.sum(res))
    number_of_hits = int(data.shape[1] - number_of_misses)
    index = (res == 1.0).reshape(-1).astype(bool)

    missed_elements = data[:, index]
    number_rows_columns = int(math.ceil(math.sqrt(min(num, number_of_misses))))

    if number_rows_columns > 0:
        figurePlotting(missed_elements, number_rows_columns, is_cifar, name + ": Misclassified")

    index = np.invert(index)
    hit_elements = data[:, index]
    number_rows_columns = int(math.ceil(math.sqrt(min(num, number_of_hits))))

    if number_rows_columns > 0:
        figurePlotting(hit_elements, number_rows_columns, is_cifar, name + ": Correct")
    plt.show()


def logreg_image(train: np.ndarray, test: np.ndarray, regularization_coefficients: List[float],
                 is_cifar: bool = False) -> None:
    '''
    without reg : 0
    with reg: regularization_coefficients = 1 / 2sigma^2
    :param train: data and labels for classifier training
    :param test: data and labels for classifier test
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    '''
    train_label = np.transpose(train[0, :].astype(np.double))
    train_label[train_label < 0] = 0.0
    train_x = train.astype(np.double)
    train_x[0, :] = 1.0

    print("Dataset ballance in training {:.2f}%".format(100 * np.sum(train_label) / len(train_label)))

    test_label = np.transpose(test[0, :].astype(np.double))
    test_label[test_label < 0] = 0.0
    test_x = test.astype(np.double)
    test_x[0, :] = 1.0

    print("Dataset ballance in test {:.2f}%".format(100 * np.sum(test_label) / len(test_label)))

    for r in regularization_coefficients:
        logreg = LOGREG(r)

        print('Training a LOGREG classifier with regularization coefficient: {}'.format(r))

        # training
        logreg.train(train_x, train_label, 50)
        print('Training')
        logreg.printClassification(train_x, train_label)
        print('Test')
        logreg.printClassification(test_x, test_label)

        visualizeClassification(train_x[1:, :], train_label, logreg.classify(train_x),
                                3 * 3, is_cifar, 'training with reg: {}'.format(r))
        visualizeClassification(test_x[1:, :], test_label, logreg.classify(test_x), 3 * 3,
                                is_cifar, 'test with reg: {}'.format(r))


def testMNIST() -> None:
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST38")
    data = scipy.io.loadmat(dataPath + 'zip38.mat')
    train = data['zip38_train']
    test = data['zip38_test']
    # print("Running MNIST13")
    # data = scipy.io.loadmat(dataPath + 'zip13.mat')
    # train = data['zip13_train']
    # test = data['zip13_test']
    logreg_image(train, test, regularization_coefficients=[0.0, 0.1, 0.5])


def test_plane() -> None:
    '''
     - Load CIFAR dataset, classes plane and no_plane
     - Train a logistic regression classifier
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running Plane vs no-plane")
    data = scipy.io.loadmat(dataPath + 'plane_no_plane.mat')
    train = data['train']
    test = data['test']
    logreg_image(train, test, regularization_coefficients=[0.0001, 0.1, 0.5], is_cifar=True)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("LOGREG exercise - MNIST Example")
    print("##########-##########-##########")
    testMNIST()
    print("\n##########-##########-##########")
    print("LOGREG exercise - CIFAR Example")
    print("##########-##########-##########")
    test_plane()
