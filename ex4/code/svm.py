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
        return K
    
    def build_kernel(self,X):
        self.K = np.dot(X,X.T)


    
    def train_M(self,X,targets):
        
        X = X.T
        
        self.N = np.shape(X)[0]
        print("hoch")
        print(self.N)
        self.build_kernel(X)
        print(self.N)
        
        print(self.K.shape)


        print("HERE")
        print(X.shape)
        print(self.K.shape)
        print(targets.shape)    
                
       # sys.exit()  
        # Assemble the matrices for the constraints
        P = targets*targets.transpose()*self.K
       
#        P = np.dot(np.dot(targets, targets.T), self.K)        
        q = -np.ones((self.N,1))
        if self.C is None:
            G = -np.eye(self.N)
            h = np.zeros((self.N,1))
        else:
            G = np.concatenate((np.eye(self.N),-np.eye(self.N)))
            h = np.concatenate((self.C*np.ones((self.N,1)),np.zeros((self.N,1))))
        A = targets.reshape(1,self.N)
        b = 0.0

        # Call the quadratic solver
        sol = cvx.solvers.qp(cvx.matrix(P),cvx.matrix(q),cvx.matrix(G),cvx.matrix(h), cvx.matrix(A), cvx.matrix(b))

        # Get the Lagrange multipliers out of the solution dictionary
        lambdas = np.array(sol['x'])
        
        print(lambdas)

        # Find the (indices of the) support vectors, which are the vectors with non-zero Lagrange multipliers
        self.sv = np.where(lambdas>self.__TOL)#[0]
        self.nsupport = len(self.sv)
        print (self.nsupport, "support vectors found" )

        # Just retain the data corresponding to the support vectors
        self.X = X[self.sv,:]
        self.lambdas = lambdas[self.sv]
        print(self.lambdas)
        self.targets = targets[self.sv]

            #self.b = np.sum(self.targets)
            #for n in range(self.nsupport):
            #self.b -= np.sum(self.lambdas*self.targets.T*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
            #self.b /= len(self.lambdas)
        #print "b=",self.b

        self.b = np.sum(self.targets)
        for n in range(self.nsupport):
            self.b -= np.sum(self.lambdas*self.targets*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
            self.b /= len(self.lambdas)
        #print "b=",self.b

        #bb = 0
        #for j in range(self.nsupport):
            #tally = 0    
            #for i in range(self.nsupport):
                #tally += self.lambdas[i]*self.targets[i]*self.K[self.sv[j],self.sv[i]]
            #bb += self.targets[j] - tally
        #self.bb = bb/self.nsupport
        #print self.bb
                
        if self.kernel == 'poly':
            def classifier(Y,soft=False):
                K = (1. + 1./self.sigma*np.dot(Y,self.X.T))**self.degree

                self.y = np.zeros((np.shape(Y)[0],1))
                for j in range(np.shape(Y)[0]):
                    for i in range(self.nsupport):
                        self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
                    self.y[j] += self.b
                
                if soft:
                    return self.y
                else:
                    return np.sign(self.y)
    
        elif self.kernel == 'rbf':
            def classifier(Y,soft=False):
                K = np.dot(Y,self.X.T)
                c = (1./self.sigma * np.sum(Y**2,axis=1)*np.ones((1,np.shape(Y)[0]))).T
                c = np.dot(c,np.ones((1,np.shape(K)[1])))
                aa = np.dot(self.xsquared[self.sv],np.ones((1,np.shape(K)[0]))).T
                K = K - 0.5*c - 0.5*aa
                K = np.exp(K/(2.*self.sigma**2))

                self.y = np.zeros((np.shape(Y)[0],1))
                for j in range(np.shape(Y)[0]):
                    for i in range(self.nsupport):
                        self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
                    self.y[j] += self.b

                if soft:
                    return self.y
                else:
                    return np.sign(self.y)
        else:
            print( "Error -- kernel not recognised")
            return

        self.classifier = classifier    


    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar
        x = x.T
        NUM = x.shape[0]

        # we'll solve the dual
        # obtain the kernel
        ## Todo update K expect for else
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
            K = np.dot(x,x.T)

        if self.C is None:
            G = -np.eye(NUM)
            h = np.zeros((NUM,1))
        else:
            print("Using Slack variables")
            ## ToDO to do update
            G = 0
            h = 0
      
        cvx.solvers.options['show_progress'] = False
        K = np.dot(x, x.T)       
        A = y.reshape(1, NUM)
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
        lambdas = np.array(solution['x'])

        ######
        # Compute below values according to the lecture slides
        self.lambdas = lambdas[lambdas>self.__TOL]  # Only save > 0

        # List of support vectors
        self.sv = np.where(lambdas>self.__TOL)[0]
        # List of labels for the support vectors (-1 or 1 for each support vector)
        self.sv_labels = y.T[self.sv]
        print(f'labels: {self.sv_labels}')

        ## reduce data to important close points
        x = x[self.sv, :]

        if kernel is None:
            # SVM weights used in the linear SVM

            w = np.linalg.inv(np.dot(x.T , x))
            w = np.dot(w, x.T)
            self.w = np.dot(w, self.sv_labels)
            print(f'w {self.w}')


            ## bias = f(x)- wT * x
            # Use the mean of all support vectors for stability when computing the bias (w_0)
            self.bias = 0 #np.sum(self.sv_labels - self.w * x )# Bias
            print(self.w.T.shape)
            print(x.shape)
            self.bias = self.sv_labels - np.dot(self.w.T, np.sum(x.T))
            print(f'Bias {self.bias}')

        ## toDO implement kernel
        else:
          self.w = 0
          # Use the mean of all support vectors for stability when computing the bias (w_0).
          # In the kernel case, remember to compute the inner product with the chosen kernel function.
          self.bias = 0 # Bias

        # TODO: Implement the KKT check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = np.sum(np.dot(self.lambdas, self.sv_labels))
        print(f'kkt {kkt2_check}')
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
        # TODO: Implement
        classified = self.classifyLinear(x)


        #print(f'classified {classified}')
        #print(f'y {y}')
        (_, N) = y.shape
        #easy solution to get error
        result = np.abs(np.sum(y-classified)/2) / N

        ## for loop for error

        result = 0
        for i in range(N):
            if (y[0][i] != classified[0][i]):
                result += 1

        result /= N

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
