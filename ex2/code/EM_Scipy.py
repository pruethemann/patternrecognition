import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

import sys, os, math
import random
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from statistics import mean 

import scipy as sp

from numpy.linalg import inv
from matplotlib.patches import Ellipse
from imageHelper import imageHelper
from myMVND import MVND
from classifyHelper import classify, get_prior

n_samples = 200

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
#shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
#print(type(shifted_gaussian))

data = scipy.io.loadmat(os.path.join('../data/', 'gmmdata.mat'))['gmmdata']
data = data.T

#print(shifted_gaussian.shape)
shifted_gaussian = data
x,y = shifted_gaussian[:,0], shifted_gaussian[:,1]
print(type(shifted_gaussian))
print(x.shape)
#plt.plot(x,y, "p")





## generate zero centered stretched Gaussian data
#C = np.array([[0., -0.7], [3.5, .7]])
#stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
#
## concatenate the two datasets into the final training set
#X_train = np.vstack([shifted_gaussian, stretched_gaussian])
##☻plt.plot(X_train)

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(shifted_gaussian)
print(clf.means_)
print()
print(clf.covariances_)
print(clf.weights_)

# display predicted scores by the model as a contour plot
x = np.linspace(-8., 9.)
y = np.linspace(-6., 9.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.plot(shifted_gaussian[:, 0], shifted_gaussian[:, 1], "p")

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()