import sys
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from pca import PCA
from scipy.spatial import distance

# TODO: Implement euclidean distance between two vectors
def euclideanDistance(a: np.ndarray, b: np.ndarray) -> float:
    '''
    :param a: vector
    :param b: vector
    :return: scalar
    '''

    distance = (a - b)**2

    return np.sqrt(np.sum(distance))


# TODO: Implement mahalanobis distance between two vectors
def mahalanobisDistance(a: np.ndarray, b: np.ndarray, invC: np.ndarray) -> float:
    '''
    :param a: vector
    :param b: vector
    :param invS: np.ndarray
    :return: scalar
    '''
    diff = a-b
    distance = (diff.T) @ invC @ diff
    return np.sqrt(distance)


def faceRecognition() -> None:
    '''
    Train PCA with with 25 components
    Project each face from 'novel' into PCA space to obtain feature coordinates
    Find closest face in 'gallery' according to:
        - Euclidean distance
        - Mahalanobis distance
    Redo with different PCA dimensionality

    What is the effect of reducing the dimensionality?
    What is the effect of different similarity measures?
    '''
    numOfPrincipalComponents = 25
    # TODO: Train a PCA on the provided face images

    # TODO: Plot the variance of each principal component - use a simple plt.plot()

    # TODO: Implement face recognition

    # TODO: Visualize some of the correctly and wrongly classified images (see example in exercise sheet)


def faceLoaderExample() -> None:
    '''
    Face loader and visualizer example code
    '''
    matgal = scipy.io.loadmat('../data/gallery.mat')
    gall = matgal['gall'][0]

    numOfFaces = gall.shape[0]
    [N, M] = gall.item(0)[1].shape

    print("NumOfFaces in dataset", numOfFaces)

    # Show first image
    plt.figure(0)
    plt.title('First face')
    n = 0
    facefirst = gall.item(n)[1]
    faceId = gall.item(n)[0][0]
    print('Face got face id: {}'.format(faceId))
    plt.imshow(facefirst, cmap='gray')

    plt.show()

def faceLoader() -> None:
    '''
    Face loader and visualizer example code
    '''

    gall = importGallery()
    print(gall.shape)

    gall = gall[:, :10]
    print(gall.shape)

    # Show first image
    plt.figure(0)
    plt.title('First face')
    n = 0
    nComponents = 10
    pca = PCA(nComponents)
    face = gall[:, :1]
    print(face.shape)

   # face = face.reshape(24576,1)
   # print(face.shape)
    mu, U, C, data = pca.train(gall)
    alpha = pca.to_pca(data)
  #  print(alpha.shape)


#    faceId = gall.item(n)[0][0]
 #   print('Face got face id: {}'.format(faceId))
    face = alpha[:, :1]
    print(face.shape)
    face = face.reshape(192,128)
    plt.imshow(face, cmap='gray')

    plt.show()

def importGallery() -> np.array:
    ## Image Data set
    mat = scipy.io.loadmat('../data/novel.mat')
    data = mat['novel']
    imageCount = 144
    featuresize = 192 * 128
    gallery = np.zeros(featuresize)
    ## Extract all images and save it in a vector
    for images in data:
        for image in images:
            ## Transform a 192 x 128 image in a single 1D vector
            imagecol = np.ravel(image[1])
            ## Horizontally stack images to multidimensional gallery
            gallery = np.vstack((gallery, imagecol))
        break
    gallery = gallery.T[ : , 1:]
    return gallery

def testing():
    """
    Test
    """
    a= np.array([2,3,-1])
    b= np.array([4,1,-2])
    S = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])

    print(mahalanobisDistance(a,b,S))
    #S = np.linalg.inv(S)
    print(distance.mahalanobis(a,b,S))

def facesDataVariance() -> None:
    '''
    Function to compute the variance of the face PCA principal components
    EXTRA: Visualize the eigenfaces
    '''


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA images!")
  #  faceLoaderExample()
    faceLoader()
    #facesDataVariance()
    #faceRecognition()
    print("##########-##########-##########")
    print("Done!")
