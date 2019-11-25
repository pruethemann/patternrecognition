import numpy as np
import matplotlib.pyplot as plt
import math
import json


class PCA():
    '''
    Principal Component Analysis
    Specify maximum number of components in the construction (__init__)
    '''

    def __init__(self, maxComponents=-1) -> None:
        self._maxComponents = maxComponents

    def plot_pca(self, X, maxxplot=200) -> None:
        """
        Plot pca data and first 2 principal component directions
        Used to visualize the toy dataset
        """
        vec1len = math.sqrt(self.C[0])
        vec2len = math.sqrt(self.C[1])
        # Take random subset from X for plotting (max 200)
        scat = X[:, np.random.permutation(np.min((maxxplot, X.shape[1])))]
        plt.scatter(scat[0, :], scat[1, :])
        plt.quiver(self.mu[0], self.mu[1], self.U[0, 0] * vec1len, self.U[0, 1] * vec1len, angles='xy',
                   scale_units='xy', scale=1)
        plt.quiver(self.mu[0], self.mu[1], self.U[1, 0] * vec2len, self.U[1, 1] * vec2len, angles='xy',
                   scale_units='xy', scale=1)
        plt.grid()
        plt.gca().axes.set_xlim([-5.5, 5.5])
        plt.gca().axes.set_ylim([-3, 8])

    def train(self, X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        Compute PCA "manually" by using SVD
        Refer to the LinearTransform slides for details
        NOTE: Remember to set svd(, full_matrices=False)!!!
        :param X: Training data
        '''
        # TODO: Implement PCA mean (mu), principal components (U) and variance (S) of the given data (X)
        self.mu = mu
        self.U = U
        self.C = C  # Covariance matrix in vector form
        return mu, U, C


    def to_pca(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: Data to be projected into PCA space
        :return: alpha - feature vector
        '''
        # TODO: Exercise 1
        return alpha

    def from_pca(self, alpha: np.ndarray) -> np.ndarray:
        '''
        :param alpha: feature vector
        :return: X in the original space
        '''
        # TODO: Exercise 1
        return Xout

    def project(self, X: np.ndarray, k: int) -> np.ndarray:
        '''
        :param X: Data to be projected into PCA space
        :param k: Dimensionality the projection should be limited to
        :return: projected data X in a k dimensional space
        '''
        # TODO: Exercise 1
        return x_projected
