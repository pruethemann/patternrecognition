import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    
    # EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean =self.calculate_mean(data) #  np.mean(data) #
        self.cov  = np.cov(data)

    # EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> np.ndarray:    
        
       return multivariate_normal.pdf(x, self.mean, self.cov)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        log = multivariate_normal.logpdf(x, self.mean, self.cov)
        return log
   
    def calculate_mean(self, data) -> np.ndarray:
        dim, n = np.shape(data)
        
        mean = np.zeros((dim,1))

        for d in range(dim):
            mean[d][0] = np.mean(data[d])            

        # Dirty hack
        test = np.zeros(3)
        test[0] = mean[0]
        test[1] = mean[1]
        test[2] = mean[2]
        mean = test
        
        return mean