import sys
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pca import PCA


def toyExample() -> None:
    mat = scipy.io.loadmat('../data/toy_data.mat')
    data = mat['toy_data']

    # TODO: Train PCA
    pca = ???

    print("Variance")
    # TODO 1.2: Compute data variance to the S vector computed by the PCA
    # TODO 1.3: Compute data variance for the projected data (into 1D) to the S vector computed by the PCA

    plt.figure()
    plt.title('PCA plot')
    plt.subplot(1, 2, 1)  # Visualize given data and principal components
    # TODO 1.1: Plot original data (hint, use the plot_pca function
    plt.subplot(1, 2, 2)
    # TODO 1.3: Plot data projected into 1 dimension
    plt.show()


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA Toy-example")
    toyExample()
    print("##########-##########-##########")
    print("Done!")
