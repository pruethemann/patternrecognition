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
        # TODO: Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
        # @pars - pass the possible kernel function parameter to this input

        ## toDo: understand
        n, m = x.shape
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = kernelFunction(x[:, i], x[:, j], pars)


        return K
    
    def build_kernel(self,X):
        self.K = np.dot(X,X.T)


    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar
        ## I somehow did it the transposed way
        #x = x.T
        NUM = x.shape[1]

        # we'll solve the dual
        # obtain the kernel
        ## Todo update K expect for else
        if kernel == 'linear':
            # TODO: Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            K = self.__computeKernelMatrix__(x, self.__linearKernel__(), None)
            self.kernel = self.__linearKernel__
        elif kernel == 'poly':
            # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__polynomialKernel__, kernelpar)
        elif kernel == 'rbf':
            # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__gaussianKernel__, kernelpar)
        else: # Toy example
            print('Fitting linear SVM')
            # TODO: Compute the kernel matrix for the linear SVM
            K = self.__computeKernelMatrix__(x, self.__linearKernel__, None)

        if self.C is None:
            G = -np.eye(NUM)
            h = np.zeros((NUM))
        else:
            print("Using Slack variables")
            ## ToDO to do update
            G = np.vstack((-identity_matrix, identity_matrix))
            h = np.hstack((np.zeros(NUM), np.ones(NUM) * self.C))
      
        cvx.solvers.options['show_progress'] = False
       # K = np.dot(x, x.T)
        #A = y.reshape(0, NUM)
        A = cvx.matrix(y)
        b = 0.0
        P = y * y.transpose() * K

        P = cvx.matrix(P)
        q = -np.ones((NUM, 1))
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(b)

        ## Execute cvx solver and retrieve all lambdas
        solution = cvx.solvers.qp(P,q,G,h,A,b)
        ## extract lambdas. Ugly trick over transpose to reduce dimensions
        lambdas = np.array(solution['x']).T[0]

        ######
        # Compute below values according to the lecture slides
        # Extract lambdas which are > 0
        self.lambdas = lambdas[lambdas>self.__TOL]

        index = np.where(lambdas>self.__TOL)[0]
        # List of support vectors. Extract sv by taking only the colums with the indices of the lambdas
        self.sv = x[:, index]
        # List of labels for the support vectors (-1 or 1 for each support vector)
        self.sv_labels = y[0, index]

        sv_count = self.sv.shape[1]

        if kernel is None:
            ### WEIGHT
            self.w = 0
            # Calculate weight. Lecture 6, Slide 25
            for i in range(sv_count):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.sv[:, i]  # SVM weights used in the linear SVM

        ## toDO implement kernel
        else:
          # In the kernel case, remember to compute the inner product with the chosen kernel function.
            self.w = 0
            for i in range(self.sv.shape[1]):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.k[:, i]

        ### BIAS
        self.bias = 0
        # get mean of all sv axis=1 sums up only rows # Use the mean of all support vectors for stability when computing the bias (w_0)
        mean = np.sum(self.sv, axis=1)

        mean = np.array([mean / self.lambdas.shape[0]]).T
        # Calculate weight. Lecture 6, Slide 25
        self.bias = self.sv_labels[0] - np.dot(np.array(self.w).T, mean)
        print(f'Bias {self.bias}')

        # check implementation with KKT2 check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = np.sum(np.dot(self.lambdas, self.sv_labels))
        #kkt2_check = np.dot(self.lambdas, self.sv_labels)
        #print(f'kkt {kkt2_check}')
        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: Array of classification values (-1.0 or 1.0)
        '''
        # Implement
        classified = np.dot(self.w.T, x) + 0 #self.bias
        classified = (classified > 0).astype(int)
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

        print(classified)

        (_, N) = y.shape
        diff = y-classified
        #print(diff)
        result = 0
        for i in range(N):
            if diff[0][i] != 0:
                result += 1
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
