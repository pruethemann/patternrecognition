import cvxopt as cvx
import numpy as np
from scipy.linalg import norm


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
        # TODO: Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
        # @pars - pass the possible kernel function parameter to this input
        return K

    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        ## i dont know what to do
        cvx.solvers.options['show_progress'] = False
        cvx.matrix()

        solution = cvx.solvers.qp(P, q, G,h,A, b)
        lambdas = solution['x']

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            # TODO: Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            K = 0
            self.kernel = self.__linearKernel__
        elif kernel == 'poly':
            # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            K = 0
        elif kernel == 'rbf':
            # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            K = 0
        else: # Toy example
            print('Fitting linear SVM')
            # TODO: Compute the kernel matrix for the linear SVM
            K = 0

        if self.C is None:
            G = 0
            h = 0
        else:
            print("Using Slack variables")
            G = 0
            h = 0


        # TODO: Compute below values according to the lecture slides
        self.lambdas = None # Only save > 0
        self.sv = None # List of support vectors
        self.sv_labels = None # List of labels for the support vectors (-1 or 1 for each support vector)
        if kernel is None:
          self.w = None # SVM weights used in the linear SVM
          # Use the mean of all support vectors for stability when computing the bias (w_0)
          self.bias = None # Bias
        else:
          self.w = None
          # Use the mean of all support vectors for stability when computing the bias (w_0).
          # In the kernel case, remember to compute the inner product with the chosen kernel function.
          self.bias = None # Bias

        # TODO: Implement the KKT check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = None
        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: Array of classification values (-1.0 or 1.0)
        '''
        # Implement
        classified = np.dot(self.w.T, x) + self.bias
        classified = (classified > 0).astype(int)
        classified = np.where(classified == 0, -1, classified)
        return classified

    def printLinearClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: Array (1D) of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement

        return 0

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))
