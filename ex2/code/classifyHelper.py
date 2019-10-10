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
    # determine fraction of image              
    prior_skin = image_mask[image_mask == 0]
    
    (skin_pixel,) = prior_skin.shape
    prior_skin = skin_pixel / (N*M)
    prior_nonskin = 1 - prior_skin
    
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
    ## convert RGB-image into a linear image with 3 dimensions for RGB and n*m colums
    im_rgb_lin = img.getLinearImage()

    if (type(skin_mvnd) != list):
        skin_mvnd = [skin_mvnd]
    if (type(notSkin_mvnd) != list):
        notSkin_mvnd = [notSkin_mvnd]
        
    ## get likelihood for every pixel being skin
    likelihood_of_skin_rgb = log_likelihood(im_rgb_lin, skin_mvnd)
        ## get likelihood for every pixel being non-skin
    likelihood_of_nonskin_rgb = log_likelihood(im_rgb_lin, notSkin_mvnd)

    # Truth: 0 for skin, 1 for non-skin
    testmask = mask.getLinearImageBinary().astype(int)[:, 0]
    npixels = len(testmask)
    
    ## Bayes classification. liklehood is positive if more proable to be skin   
    likelihood_rgb = likelihood_of_skin_rgb - likelihood_of_nonskin_rgb    
    ## Classification. If value larger than 0 it's skin. And get's classified with 1
    skin = (likelihood_rgb > 0).astype(int)  
    
    imgMinMask = skin - testmask

    # EXERCISE 2 - Error Rate without prior
    fp = 0 # false positive. Pixels classified as skin (1) but should be nonskin
    fn = 0 # false negative. Pixel classified as non-skin (0) but should be skin
    totalError = 0

    ## check every pixel
    for i in range(npixels):
        # It's skin but predicted as non-skin: false negative
        if testmask[i] == 1 and int(skin[i]) == 0:  
            fn += 1
        # It's non-skin but predicted as skin: false positive
        if testmask[i] == 0 and int(skin[i]) == 1: 
            fp += 1            
    
    totalError = (fp + fn)

    print('----- ----- -----')
    print('Total Error WITHOUT Prior =', totalError, " ", round(totalError/npixels,3))
    print('false positive rate =', fp, " ",  round(fp/npixels,3))
    print('false negative rate =', fn, " ", round(fn/npixels,3))

    # TODO: EXERCISE 2 - Error Rate with prior
    
    ## Marginal
    likelihood_rgb_with_prior = 0
#    
    for i in range(npixels):
        likelihood_rgb_with_prior += np.exp(likelihood_rgb[0]) * prior_skin
#    
    skin_prior = prior_skin * np.exp(likelihood_of_skin_rgb) / likelihood_rgb_with_prior
    #♥skin_prior = prior_skin * np.exp(likelihood_of_skin_rgb) / prior_nonskin
#    print(likelihood_rgb_with_prior)
    # fix
    imgMinMask_prior = skin_prior - testmask
    fp_prior = 0
    fn_prior = 0
    totalError_prior = 0
    
    ## check every pixel
    for i in range(npixels):
        # It's skin but predicted as non-skin
        if testmask[i] == 1 and int(skin_prior[i]) == 0:  
            fn_prior += 1
        # It's non-skin but predicted as skin
        if testmask[i] == 0 and int(skin_prior[i]) == 1: 
            fp_prior += 1            
    
    totalError_prior = fp_prior + fn_prior       
    
    print('----- ----- -----')
    print('Total Error WITH Prior =', totalError_prior, " ", totalError_prior/npixels)
    print('false positive rate =', fp_prior, " ", fp_prior/npixels)
    print('false negative rate =', fn_prior, " ", fn_prior/npixels)
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
    plt.title('FalsePositive Without Prior')
    plt.subplot(4, 5, 10)
    plt.imshow(fnImage, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative Without Prior')
    plt.subplot(4, 5, 15)
    plt.imshow(fpImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive PRIOR')
    plt.subplot(4, 5, 20)
    plt.imshow(fnImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative PRIOR')
