import sys
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pca import PCA
from sklearn import datasets


def importGallery() -> np.array:
    ## Image Data set
    mat = scipy.io.loadmat('../data/gallery.mat')
    data = mat['gall']
    imageCount = 144
    featuresize = 192 * 128
    gallery = np.zeros(featuresize)
    ## Extract all images and save it in a vector
    for images in data:
        for image in images:
            ## Transform a 192 x 128 image in a single 1D vector
            imagecol = np.ravel(image[1])
            ## Horizontally stack images to multidimensional gallery
            gallery = np.vstack((gallery, imagecol))
        break
    gallery = gallery.T[ : , 1:]
    return gallery


def toyExample() -> None:
    ## Toy Data Set
    mat = scipy.io.loadmat('../data/toy_data.mat')
    data = mat['toy_data']

    #data = importGallery()
    #data = data[ : , :1]

    ## Iris dataset. Just for testing purposes
    #iris = datasets.load_iris()
    #data = iris['data'].astype(np.float32)  # a 150x4 matrix with features
    #data = data.T
    # TODO: Train PCA
    nComponents = 2

    pca = PCA(nComponents)

    ## 1.1 Calculate PCA manuelly. SVD is following
    #pca.pca_manuel(data)
    ## 1.2 Calculate PCA via SVD
    mu, U, C, dataCenter = pca.train(data)

    ## 2. Transform RAW data using first n principal components
    alpha = pca.to_pca(dataCenter)

    ## 3. Backtransform alpha to Raw data
    Xout = pca.from_pca(alpha)

    print("Variance")
    # TODO 1.2: Compute data variance to the eigenvalue vector computed by the PCA

    print(f'Total Variance: {np.var(data)}')
    print(f'Eigenvalues: {C} \n')

    # TODO 1.3: Compute data variance for the projected data (into 1D) to the S vector computed by the PCA
    print(f'Total Variance Transform: {np.var(alpha)}')
    print(f'Mean Eigenvalues: {np.mean(C)}')

    ## Plot only if fewer than 2 components
    if nComponents == 2:
        plt.figure()
        plt.title('PCA plot')
        plt.subplot(1, 2, 1)  # Visualize given data and principal components
        # TODO 1.1: Plot original data (hint, use the plot_pca function
        pca.plot_pca(data)
        plt.subplot(1, 2, 2)
        # TODO 1.3: Plot data projected into 1 dimension
        pca.plot_pca(Xout)
        plt.show()


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA Toy-example")
    toyExample()
    print("##########-##########-##########")
    print("Done!")
