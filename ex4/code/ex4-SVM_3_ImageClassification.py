import matplotlib
import sys
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from svm import SVM

matplotlib.use('TkAgg')

dataPath = '../data/'


def get_rbg_image(image: np.ndarray) -> np.ndarray:
    img = np.zeros((32, 32, 3))
    img[:, :, 0] = np.reshape(image[:1024], (32, 32))
    img[:, :, 1] = np.reshape(image[1024:2048], (32, 32))
    img[:, :, 2] = np.reshape(image[2048:], (32, 32))

    return img


def figurePlotting(imgarray: np.ndarray, N: int, is_cifar: bool, name: str = '', random: bool = True) -> None:
    '''
    CIFAR / MNIST image visualization - rescaling the vector images to 32x32 and visualizes in a matplotlib plot
    :param imgarray: Array of images to be visualized, each column is an image
    :param N: Number of images per row/column
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


def visualizeClassification(data: np.ndarray, labels: np.ndarray, predictions: np.ndarray, num: int, is_cifar: bool,
                            name='') -> None:
    '''
    Use SVM classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: CIFAR data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of CIFAR images to show
    :param name: Optional name of the plot
    '''
    res = np.abs(predictions - labels) / 2.0
    number_of_misses = int(np.sum(res))
    number_of_hits = int(data.shape[1] - number_of_misses)
    index = (res == 1.0).reshape(-1).astype(bool)

    missed_vectors = data[:, index]
    n_pictures = int(math.ceil(math.sqrt(min(num, number_of_misses))))

    if n_pictures > 0:
        figurePlotting(missed_vectors, n_pictures, is_cifar, name + ": Misclassified")

    index = np.invert(index)
    hit_vectors = data[:, index]
    n_pictures = int(math.ceil(math.sqrt(min(num, number_of_hits))))

    if n_pictures > 0:
        figurePlotting(hit_vectors, n_pictures, is_cifar, name + ": Correct")
    plt.show()


def svm_image(train: np.ndarray, test: np.ndarray, is_cifar: bool) -> SVM:
    '''
    Train an SVM with the given training data and print training + test error
    :param train: Training data
    :param test: Test data
    :return: Trained SVM object
    '''

    linear = True

    _, N = np.shape(train)
    if is_cifar:
        # Adapt to the size of the training set to gain speed or precision
        N = 200

    train_label = np.reshape(train[0, :N], (1, N)).astype(float)
    train_label[train_label == 0] = -1.0
    train_x = train[1:, :N].astype(float)

    _, n = np.shape(test)
    test_label = np.reshape(test[0, :], (1, n)).astype(float)
    test_label[test_label == 0] = -1.0
    test_x = test[1:, :].astype(float)

    # TODO: Train svm
    svm = ???
    # # use a linear instead of a linear kernel to improve speed
    # linear = ??? # bool

    print("Training error")
    # TODO: Compute training error of SVM

    print("Test error")
    # TODO: Compute test error of SVM

    if linear:
        visualizeClassification(train_x, train_label, svm.classifyLinear(train_x), 3 * 3, is_cifar, 'training')
        visualizeClassification(test_x, test_label, svm.classifyLinear(test_x), 3 * 3, is_cifar, 'test')
    else:
        visualizeClassification(train_x, train_label, svm.classifyKernel(train_x), 3 * 3, is_cifar, 'training')
        visualizeClassification(test_x, test_label, svm.classifyKernel(test_x), 3 * 3, is_cifar, 'test')
    return svm


def test_ship() -> None:
    '''
     - Load CIFAR dataset, classes ship and no_ship
     - Train a linear or kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running Ship or no-ship")
    toy = scipy.io.loadmat(dataPath + 'ship_no_ship.mat')
    train = toy['train']
    test = toy['test']
    svm_image(train, test, is_cifar=True)


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
    svm_image(train, test, is_cifar=False)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("SVM exercise - MNIST Example")
    print("##########-##########-##########")
    testMNIST()
    print("\n##########-##########-##########")
    print("SVM exercise - CIFAR Example")
    print("##########-##########-##########")
    test_ship()
