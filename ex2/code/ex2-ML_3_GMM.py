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

matplotlib.use('TkAgg')

dataPath = '../data/'


def gmm_draw(gmm, data, plotname='') -> None:
    '''
    gmm helper function to visualize cluster assignment of data
    :param gmm:         list of MVND objects
    :param data:        Training inputs, #(dims) x #(samples)
    :param plotname:    Optional figure name
    '''
    plt.figure(plotname)
    K = len(gmm)
    N = data.shape[1]
    dists = np.zeros((K, N))
    for k in range(0, K):
        d = data - (np.kron(np.ones((N, 1)), gmm[k].mean)).T
        dists[k, :] = np.sum(np.multiply(np.matmul(inv(gmm[k].cov), d), d), axis=0)
    comp = np.argmin(dists, axis=0)

    # plot the input data
    ax = plt.gca()
    ax.axis('equal')
    for (k, g) in enumerate(gmm):
        indexes = np.where(comp == k)[0]
        kdata = data[:, indexes]
        g.data = kdata
        ax.scatter(kdata[0, :], kdata[1, :])

        [_, L, V] = scipy.linalg.svd(g.cov, full_matrices=False)
        phi = math.acos(V[0, 0])
        if float(V[1, 0]) < 0.0:
            phi = 2 * math.pi - phi
        phi = 360 - (phi * 180 / math.pi)
        center = np.array(g.mean).reshape(1, -1)

        d1 = 2 * np.sqrt(L[0])
        d2 = 2 * np.sqrt(L[1])
        ax.add_patch(Ellipse(center.T, d1, d2, phi, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1, fill=False))
        plt.plot(center[0, 0], center[0, 1], 'kx')


def gmm_em(data, K: int, iter: int, plot=False) -> list:
    '''
    EM-algorithm for Gaussian Mixture Models
    Usage: gmm = gmm_em(data, K, iter)
    :param data:    Training inputs, #(dims) x #(samples)
    :param K:       Number of GMM components, integer (>=1)
    :param iter:    Number of iterations, integer (>=0)
    :param plot:    Enable/disable debugging plotting
    :return:        List of objects holding the GMM parameters.
                    Use gmm[i].mean, gmm[i].cov, gmm[i].c
    '''
    eps = sys.float_info.epsilon
    [d, N] = data.shape

    # TODO: EXERCISE 2 - Implement E and M step of GMM algorithm
    # 1. Hint - first randomly assign a cluster to each sample
    
    ## randomly assign 3 different means between -4 - 8 and -4 and 4
    gmm = []
    liklehood = []
    loglike= 0
    for k in range(K):
        c = MVND(data)
        c.mean[0] = random.uniform(-4,4)
        c.mean[1] = random.uniform(-4,4)
        c.c = 1/3
        gmm.append(c)
        
    gmm_draw(gmm,data, "TOY")
    plt.show()
    # Hint - then iteratively update mean, cov and p value of each cluster via EM
    for i in range(iter):     
        likelihood, wp1, wp2, wp3 = e_step(loglike, data, N, gmm)
        #m_step(gmm, wp1, wp2, wp3)
        clusters = optimize(wp1,wp2,wp3)
        gmm = calculate_mean(gmm,clusters, data)
        if(plot):
            gmm_draw(gmm,data, "TOY")
            plt.show()
    
    ## shows means
    print("Means:")
    for i,g in enumerate(gmm):
        print("Cluster ", i, ": ", g.mean)
        print("Cov: \n", g.cov)
    return gmm

def e_step(loglike, data, N, gmm):
    ## extract X
    
    for n in range(1):
        x = np.transpose(data)

        c1 = gmm[0].pdf(x) * gmm[0].c
        c2 = gmm[1].pdf(x)  * gmm[1].c
        c3 = gmm[2].pdf(x) * gmm[2].c
        den = c1 + c2 + c3

        c1 /= den
        c2 /= den
        c3 /= den

        loglike += np.log(c1 + c2 + c3)

    return loglike, c1, c2, c3

 
def optimize(c1,c2,c3):
    clusters = np.zeros(200)
    for i in range(200):
        if c1[i] > c2[i] and c1[i] > c3[i]:
            decision = 0
        elif c2[i] > c1[i] and c2[i] > c3[i]:
            decision = 1
        else:
            decision = 2
        
        clusters[i] = decision
    return clusters
        
def calculate_mean(gmm,clusters, data):

    c1 = np.where(clusters==0)[0] # Get indexes
    c2 = np.where(clusters==1)[0] # Get indexes
    c3 = np.where(clusters==2)[0] # Get indexes

    (N1,) = c1.shape
    (N2,) = c2.shape
    (N3,) = c3.shape    
    sum1 = np.zeros((2,N1))
    sum2 = np.zeros((2,N2))    
    sum3 = np.zeros((2,N3))

    for col in range(N1):
        index = c1[col]
        sum1[0][col] = data[0][index]
        sum1[1][col] = data[1][index]
        
    for col in range(N2):
        index = c2[col]
        sum2[0][col] = data[0][index]
        sum2[1][col] = data[1][index]
        
        
    for col in range(N3):
        index = c3[col]
        sum3[0][col] = data[0][index]
        sum3[1][col] = data[1][index]

    gmm[0].mean = gmm[0].calculate_mean(sum1)
    gmm[1].mean = gmm[1].calculate_mean(sum2)    
    gmm[2].mean = gmm[2].calculate_mean(sum3)    

    gmm[0].cov = np.cov(sum1)
    gmm[1].cov = np.cov(sum2)
    gmm[2].cov = np.cov(sum3)    
    return gmm
    

def gmmToyExample() -> None:
    '''
    GMM toy example - load toyexample data and visualize cluster assignment of each datapoint
    '''
    gmmdata = scipy.io.loadmat(os.path.join(dataPath, 'gmmdata.mat'))['gmmdata']
    gmm_em(gmmdata, 3, 20, plot=True)


def gmmSkinDetection() -> None:
    '''
    Skin detection - train a GMM for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    K = 3
    iter = 50
    sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']
    gmms = gmm_em(sdata, K, iter)
    gmmn = gmm_em(ndata, K, iter)

    print("TRAINING DATA")
    trainingmaskObj = imageHelper()
    trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    trainingimageObj = imageHelper()
    trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    prior_skin, prior_nonskin = get_prior(trainingmaskObj)
    classify(trainingimageObj, trainingmaskObj, gmms, gmmn, "training", prior_skin=prior_skin,
             prior_nonskin=prior_nonskin)

    print("TEST DATA")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test.png'))
    classify(testimageObj, testmaskObj, gmms, gmmn, "test", prior_skin=prior_skin, prior_nonskin=prior_nonskin)

    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nMVND exercise - Toy example")
    print("##########-##########-##########")
    gmmToyExample()
    print("\nMVND exercise - Skin detection")
    print("##########-##########-##########")
    #gmmSkinDetection()
    print("##########-##########-##########")
