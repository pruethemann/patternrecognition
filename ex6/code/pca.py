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
        ## Plots eigenvectors U
        plt.quiver(self.mu[0], self.mu[1], self.U[0, 0] * vec1len, self.U[0, 1] * vec1len, angles='xy',
                   scale_units='xy', scale=1)
        plt.quiver(self.mu[0], self.mu[1], self.U[1, 0] * vec2len, self.U[1, 1] * vec2len, angles='xy',
                   scale_units='xy', scale=1)
        plt.grid()
        plt.gca().axes.set_xlim([-5.5, 5.5])
        plt.gca().axes.set_ylim([-3, 8])


    def pca_manuel(self, X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        Compute PCA "manually" by using SVD
        Refer to the LinearTransform slides for details
        NOTE: Remember to set svd(, full_matrices=False)!!!
        :param X: Training data
        '''

        ## 1. zero center data along dimension. Here features are in rows
        self.mu = mu = np.mean(X, axis=1)
        dims, ncols = X.shape

        ### ToDo: optimize
        for dim in range(dims):
            for col in range(ncols):
                X[dim][col] -= mu[dim]

        ## 2. Determine covariance matrix
        covariance = np.cov(X, rowvar=True) ## features in rows

        ## 3. Determine eigenvalues and eigenvectors of covariance matrix
        eigenv, eigenvectors = np.linalg.eig(covariance)

        ## 4. sort eigenvalues in descending order
        idx = eigenv.argsort()[::-1]

        ## 5. Sort eigenvalues and eigenvectors with index
        C, U = eigenv[idx], eigenvectors[:, idx]
        self.C, self.U = C, U
        return mu, U, C

    def train(self, X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        Compute PCA "manually" by using SVD
        Refer to the LinearTransform slides for details
        NOTE: Remember to set svd(, full_matrices=False)!!!
        :param X: Training data
        '''

        X = X.T
        self.mu = mu = np.mean(X, axis=0)
        n, p = X.shape
        # Let us assume that it is centered
        X -= mu

        # we now perform singular value decomposition of X
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S = np.diag(S)
        self.U = U
        self.C = S

        # a matrix of eigenvectors (each column is an eigenvector)
      #  print("Vectors = \n", U)
        print("lambda = \n", S)

        V = Vt.T
        Sigma = np.diag(S)
        if self._maxComponents == -1:
            self.C = C = Sigma
        else:
            self.C = C = Sigma[:self._maxComponents]


#        C = principal_components[:, 0:self._maxComponents]
#        US_k = U[:, 0:self._maxComponents].dot(S[0:self._maxComponents, 0:self._maxComponents])
        return mu, U, C


    def to_pca(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: Data to be projected into PCA space. Variables are in row
        :return: alpha - feature vector
        '''
        ## 6. Limit principal components
        if self._maxComponents > -1:
            U = self.U[:, 0:self._maxComponents]

        ## Eigenvectors in rows dot data in variables in rows
        alpha = U.T @ X
        return alpha

    def from_pca(self, alpha: np.ndarray) -> np.ndarray:
        '''
        :param alpha: feature vector
        :return: X in the original space
        '''
        ## 6. Limit principal components
        if self._maxComponents > -1:
            U = self.U[:, 0:self._maxComponents]
        ## Perform back transformation: Restored Data = limited eigenvectors (transposed) x PCA transposed (dimension in rows=
        ## eigenvectors are actually inversed but it's a orthogonal matrix
        Xout = U @ alpha
        return Xout

    def project(self, X: np.ndarray, k: int) -> np.ndarray:
        '''
        :param X: Data to be projected into PCA space
        :param k: Dimensionality the projection should be limited to
        :return: projected data X in a k dimensional space
        '''

        self._maxComponents = k

        ## 1. Calculate PCA manuelly. SVD is following
        self.pca_manuel(X)

        ## 2. Transform RAW data using first n principal components
        alpha = self.to_pca(X)

        ## 3. Backtransform alpha to Raw data
        x_projected = self.from_pca(alpha)

        return x_projected
