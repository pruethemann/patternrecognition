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
    nComponents = 2
    pca = PCA(nComponents)

    ## 1. Calculate PCA manuelly. SVD is following
    mu, U, C = pca.pca_manuel(data)

    ## zero center data
    dims, ncols = data.shape

    print(data[0, :10])
    ### ToDo: optimize
    for dim in range(dims):
        for col in range(ncols):
            data[dim][col] -= mu[dim]

    ## 2. Transform RAW data using first n principal components
    alpha = pca.to_pca(data)
    print(f'alpha: {alpha.shape}')

    ## 3. Backtransform alpha to Raw data
    Xout = pca.from_pca(alpha)

    print(Xout[0, :10])

    print("Variance")
    # TODO 1.2: Compute data variance to the S vector computed by the PCA

    print(f'Total Variance: {np.var(data)}')
    print(f'Lambdas: {np.mean(pca.C)}')

    # TODO 1.3: Compute data variance for the projected data (into 1D) to the S vector computed by the PCA
    print(f'Total Variance Transform: {np.var(alpha)}')
    print(f'Total Variance: {pca.C}')

    plt.figure()
    plt.title('PCA plot')
    plt.subplot(1, 2, 1)  # Visualize given data and principal components
    # TODO 1.1: Plot original data (hint, use the plot_pca function
    pca.plot_pca(data)
    plt.subplot(1, 2, 2)
    # TODO 1.3: Plot data projected into 1 dimension
    pca.plot_pca(pca.project(data,1))
    plt.show()


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA Toy-example")
    toyExample()
    print("##########-##########-##########")
    print("Done!")
