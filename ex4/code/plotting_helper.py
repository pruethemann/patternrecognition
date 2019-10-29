import numpy as np
import matplotlib.pyplot as plt
from svm import SVM


def plot_data(plt: plt, x: np.ndarray, y: np.ndarray, STYLE, label: str = ''):
    '''
    Visualize 2D data items - color according to their class
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param x: 2D data
    :param y: Data labels
    :param STYLE: Marker style and color in list format, ex: [['red', '+'], ['blue', '_']]
    :param label: Obtional plot name
    '''
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[:, y[0, :] == unique[li]]
        plt.scatter(x_sub[0, :], x_sub[1, :], c=STYLE[li][0], marker=STYLE[li][1], label=label + str(li))
    plt.legend()


def plot_linear_separator(plt: plt, svm: SVM, datamin: int, datamax: int):
    '''
    Visualize linear SVM separator with margins
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    '''
    x = np.arange(datamin, datamax + 1.0)
    MARG = -(svm.w[0] * x + svm.bias) / svm.w[1]
    YUP = (1 - svm.w[0] * x - svm.bias) / svm.w[1]  # Margin
    YLOW = (-1 - svm.w[0] * x - svm.bias) / svm.w[1]  # Margin
    plt.plot(x, MARG, 'k-')
    plt.plot(x, YUP, 'k--')
    plt.plot(x, YLOW, 'k--')
    for sv in svm.sv:
        plt.plot(sv[0], sv[1], 'kx')


def plot_kernel_separator(plt: plt, svm: SVM, datamin: float, datamax: float, h: float = 0.05, alpha: float = 0.25):
    '''
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    :param h: Density of classified background points
    :return:
    '''
    # function visualizes decision boundaries using color plots
    # creating meshgrid for different values of features
    xx, yy = np.meshgrid(np.arange(datamin, datamax, h), np.arange(datamin, datamax, h))
    # extracting predictions at different points in the mesh
    some = np.transpose(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.classifyKernel(some)
    Z = Z.reshape(xx.shape)
    # plotting the mesh
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=alpha)
    for sv in svm.sv:
        plt.plot(sv[0], sv[1], 'kx')
    plt.grid()
