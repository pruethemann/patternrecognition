import cvxopt as cvx
import numpy as np
from scipy.linalg import norm
import sys


class SVM(object):
    '''
    SVM class
    '''

    def __init__(self, C=None):
        self.C = C
        self.__TOL = 1e-5

    def __linearKernel__(self, x1: np.ndarray, x2: np.ndarray, _) -> float:
        # Implement linear kernel function
        # @x1 and @x2 are vectors
        return np.dot(x1.T, x2)

    def __polynomialKernel__(self, x1: np.ndarray, x2: np.ndarray, p: int) -> float:
        # Implement polynomial kernel function
        # @x1 and @x2 are vectors
        return (np.dot(x1.T, x2) + 1)**p

    def __gaussianKernel__(self, x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        # Implement gaussian kernel function
        # @x1 and @x2 are vectors
        return np.exp( - norm(x1 - x2)**2 / (2*sigma**2)  )

    def __computeKernelMatrix__(self, x: np.ndarray, kernelFunction, pars) -> np.ndarray:
        # Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input (Closures)
        # @pars - pass the possible kernel function parameter to this input

        _, M = x.shape
        K = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                K[i, j] = kernelFunction(x[:, i], x[:, j], pars)

        return K
    
    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        dim, NUM = x.shape

        # we'll solve the dual, obtain the kernel
        if kernel == 'linear':
            # Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            K = self.__computeKernelMatrix__(x, self.__linearKernel__, None)
            self.kernel = self.__linearKernel__

        elif kernel == 'poly':
            # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__polynomialKernel__, kernelpar)
            self.kernel = self.__polynomialKernel__

        elif kernel == 'rbf':
            # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__gaussianKernel__, kernelpar)
            self.kernel = self.__gaussianKernel__

        else: # Toy example No kernel
            print('Fitting linear SVM')
            # Compute the kernel matrix for the linear SVM
            K = self.__computeKernelMatrix__(x, self.__linearKernel__, None)
            self.kernel = self.__linearKernel__

        if self.C is None:
            G = -np.eye(NUM)
            h = np.zeros((NUM))

        ## Soft margin. L6 Slide 56
        else:
            print("Using Slack variables")
            I = np.eye(NUM) ## identity
            ## merge two idenitiy matrices
            G = np.vstack((-I, I))
            ## Get a matrix with the two limits 0 and C
            h = np.hstack((np.zeros(NUM), np.ones(NUM) * self.C))

        ## Calculate all matrices for cvx.solver
        P = y * y.transpose() * K ##inner product * Kernel
        A = cvx.matrix(y)
        q = -np.ones(NUM)
        b = 0.0

        ## transform to matrix
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(b)

        ## Execute cvx solver and retrieve all lambdas
        cvx.solvers.options['show_progress'] = False
        solution = cvx.solvers.qp(P,q,G,h,A,b)
        ## extract lambdas. Ugly trick over transpose to reduce dimensions
        lambdas = np.array(solution['x']).T[0]

        # Compute below values according to the lecture slides
        # Extract lambdas which are > 0
        self.lambdas = lambdas[lambdas>self.__TOL]

        index = np.where(lambdas>self.__TOL)[0]
        # List of support vectors. Extract sv by taking only the colums with the indices of the lambdas
        self.sv = x[:, index]

        # List of labels for the support vectors (-1 or 1 for each support vector)
        self.sv_labels = y[:, index]

        ## Flatten array to avoid dimension problems
        self.sv_labels = np.ravel(self.sv_labels)

        sv_count = self.sv.shape[1]
        print(f'Amount of SV {sv_count}')

        if kernel is None:
            # Calculate weight. Lecture 6, Slide 50
            self.w = 0
            for i in range(sv_count):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.sv[:, i]   # self.sv[:, i] is colum at position i

            ### BIAS. Found in ML book
            self.bias = 0
            for i in range(sv_count):
                self.bias += np.dot(self.lambdas[i] , self.sv_labels[i])

        else: # Kernel
            # In the kernel case, remember to compute the inner product with the chosen kernel function.
            self.w = 0 ## L 8, S.34
            for i in range(sv_count):
                self.w += self.lambdas[i] * self.sv_labels[i] * K[index[i], index]
            # Use the mean of all support vectors for stability when computing the bias (w_0).
            # In the kernel case, remember to compute the inner product with the chosen kernel function.
            self.bias = 0
            for i in range(sv_count):
                ## keep inner prodcut
                self.bias -= np.sum(self.lambdas * self.sv_labels * K[index[i], index])

            ## add sum of all labels and normalize
            self.bias  = (self.bias + np.sum(self.sv_labels) ) / sv_count

        # check implementation with KKT2 check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = np.sum(np.dot(self.lambdas, self.sv_labels))
        #print(f'kkt {kkt2_check}')
        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: Array of classification values (-1.0 or 1.0)
        '''

        ## classifier according to linear w.T * x + w0 >/< 0
        classified = np.dot(self.w.T, x) + self.bias

        # binary classifcation in 0 / 1
        classified = (classified > 0).astype(int)

        ## Replace all values 0 -> -1 , keep the rest
        classified = np.where(classified == 0, -1, classified)

        return classified

    def printLinearClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # Implement
        classified = self.classifyLinear(x)

        (_, N) = y.shape
        diff = y-classified

        misclassified = np.sum(np.abs(diff))
        result = 100.0 / N * misclassified
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: Array (1D) of classification values (-1.0 or 1.0)
        '''

        dim, NUM = x.shape
        classified = np.zeros(NUM)

        for i in range(NUM):
            threshold = 0
            for j in range(len(self.lambdas)):
                threshold += (self.lambdas[j] * self.sv_labels[j] * self.kernel(x[:, i], self.sv[:, j], self.kernelpar))

            ## determine border for classification at border of threshold + bias
            classified[i] = np.sign(threshold + self.bias)

        return classified

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        classified = self.classifyKernel(x)

        (_, N) = y.shape
        ## solve dimension problems
        y = np.ravel(y)
        diff = y-classified

        # result = 0
        # for i in range(N):
        #     if int(diff[i]) != 0:
        #         result += 1

        misclassified = np.sum(np.abs(diff))
        result = 100.0 / N * misclassified

        print("Total error: {:.2f}%".format(result))
