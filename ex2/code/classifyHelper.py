import math
import numpy as np
import matplotlib.pyplot as plt
from imageHelper import imageHelper
from myMVND import MVND
from typing import List
## own import
from scipy.spatial import distance


def log_likelihood(data: np.ndarray, gmm: List[MVND]) -> np.ndarray:
    '''
    Compute the likelihood of each datapoint
    Hint: Use logpdf and add likelihoods instead of multiplication of pdf's
    :param data:    Training inputs, #(samples) x #(dim)
    :param gmm:     List of MVND objects
    :return:        Likelihood of each data point
    '''
    likelihood = np.zeros((1, data.shape[0]))
    likelihood = 0
    
    N, M = np.shape(data)
    N = data.shape[0]

    # TODO: EXERCISE 2 - Compute likelihood of data
    # Note: For MVGD there will only be 1 item in the list
    for g in gmm:
        likelihood = g.logpdf(data)

    return likelihood


def get_prior(mask: imageHelper) -> (float, float):
    [N, M] = mask.shape
    image_mask = mask.image[:]
    # EXERCISE 2 - Compute the skin 0 and nonskin 1 prior

    # count all skin pixels
    skin = 0
    for n in range(N):
        for m in range(M):
            if image_mask[n][m] == 0:
                skin += 1
    
    # determine fraction of image        
    pixels_count = N * M
    prior_skin = skin/pixels_count
    prior_nonskin = (pixels_count - skin) / pixels_count

    return prior_skin, prior_nonskin


def classify(img: imageHelper, mask: imageHelper, skin_mvnd: List[MVND], notSkin_mvnd: List[MVND], fig: str = "",
             prior_skin: float = 0.5, prior_nonskin: float = 0.5) -> None:
    '''
    :param img:             imageHelper object containing the image to be classified
    :param mask:            imageHelper object containing the ground truth mask
    :param skin_mvnd:       MVND object for the skin class
    :param notSkin_mvnd:    MVND object for the non-skin class
    :param fig:             Optional figure name
    :param prior_skin:      skin prior, float (0.0-1.0)
    :param prior_nonskin:   nonskin prior, float (0.0-1.0)
    '''
    im_rgb_lin = img.getLinearImage()
    if (type(skin_mvnd) != list):
        skin_mvnd = [skin_mvnd]
    if (type(notSkin_mvnd) != list):
        notSkin_mvnd = [notSkin_mvnd]
    likelihood_of_skin_rgb = log_likelihood(im_rgb_lin, skin_mvnd)
    likelihood_of_nonskin_rgb = log_likelihood(im_rgb_lin, notSkin_mvnd)

    testmask = mask.getLinearImageBinary().astype(int)[:, 0]
    npixels = len(testmask)

    likelihood_rgb = likelihood_of_skin_rgb - likelihood_of_nonskin_rgb
    skin = (likelihood_rgb > 0).astype(int)
    
    
    imgMinMask = skin - testmask
    # TODO: EXERCISE 2 - Error Rate without prior
    fp = 0
    fn = 0
    totalError = 0

    print('----- ----- -----')
    print('Total Error WITHOUT Prior =', totalError)
    print('false positive rate =', fp)
    print('false negative rate =', fn)

    # TODO: EXERCISE 2 - Error Rate with prior
    likelihood_rgb_with_prior = 0
    skin_prior = 0
    imgMinMask_prior = 0
    fp_prior = 0
    fn_prior = 0
    totalError_prior = 0
    print('----- ----- -----')
    print('Total Error WITH Prior =', totalError_prior)
    print('false positive rate =', fp_prior)
    print('false negative rate =', fn_prior)
    print('----- ----- -----')

    N = mask.N
    M = mask.M
    fpImage = np.reshape((imgMinMask > 0).astype(float), (N, M))
    fnImage = np.reshape((imgMinMask < 0).astype(float), (N, M))
    fpImagePrior = np.reshape((imgMinMask_prior > 0).astype(float), (N, M))
    fnImagePrior = np.reshape((imgMinMask_prior < 0).astype(float), (N, M))
    prediction = imageHelper()
    prediction.loadImage1dBinary(skin, N, M)
    predictionPrior = imageHelper()
    predictionPrior.loadImage1dBinary(skin_prior, N, M)

    plt.figure(fig)
    plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    plt.imshow(img.image)
    plt.axis('off')
    plt.title('Test image')

    plt.subplot2grid((4, 5), (0, 2), rowspan=2, colspan=2)
    plt.imshow(prediction.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction')

    plt.subplot2grid((4, 5), (2, 2), rowspan=2, colspan=2)
    plt.imshow(predictionPrior.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction PRIOR')

    plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    plt.imshow(mask.image, cmap='gray')
    plt.axis('off')
    plt.title('GT mask')

    plt.subplot(4, 5, 5)
    plt.imshow(fpImage, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive')
    plt.subplot(4, 5, 10)
    plt.imshow(fnImage, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative')
    plt.subplot(4, 5, 15)
    plt.imshow(fpImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive PRIOR')
    plt.subplot(4, 5, 20)
    plt.imshow(fnImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative PRIOR')
