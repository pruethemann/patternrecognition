import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    
    # EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = self.calculate_mean(data) #  np.mean(data) #
        self.cov  = np.cov(data)

    # EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> np.ndarray:            
       return multivariate_normal.pdf(x, self.mean, self.cov)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return multivariate_normal.logpdf(x, self.mean, self.cov)
   
    ## calculates mean for every dimension / RGB
    def calculate_mean(self, data) -> np.ndarray:
        N,s = data.shape
     
        mean = np.zeros(N)
        for dim in range(N):
            mean[dim] = np.mean(data[dim])        
    
        return mean