import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = self.calculate_mean(data)
        self.cov  = np.cov(data)

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> float:       
       return multivariate_normal.pdf(x, self.mean, self.cov)

    def logpdf(self, x: np.ndarray) -> float:
       return multivariate_normal.logpdf(x, self.mean, self.cov)
   
    def calculate_mean(self, data) -> np.ndarray:
        ds, n = np.shape(data)
        
        mean = np.zeros((ds,1))
        for d in range(ds):
            mean[d] = np.mean(data[d])
            
        print(np.mean(data))
        
        print(mean)
            
        return mean


        

