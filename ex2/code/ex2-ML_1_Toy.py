import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as scio
import math

from matplotlib.patches import Ellipse

matplotlib.use('TkAgg')

dataPath = '../data/'


# Manual implementation of maximum likelihood for a 2D toy dataset

def calculate_sample_mean(data: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the sample mean of a data array (2D vector output)
    mean = np.array([[0.0],[0.0]])    
    d, n = np.shape(data)

    for c in range(n):
        mean[0] += data[0][c]
        mean[1] += data[1][c]

    mean = mean / n
    return mean


def calculate_sample_covariance_matrix(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the sample covariance matrix
    d, n = np.shape(data)
    
    ## Init variance and difference x - mu for every both datasets
    sigma1 = 0
    sigma2 = 0
    diff1 = 0
    diff2 = 0
    covariance = 0

    for c in range(n):
        diff1 = data[0][c] - mean[0]
        diff2 = data[1][c] - mean[1]
        covariance += diff1 * diff2   ## sum together the product between differences in dataset from mu
        sigma1 += diff1**2
        sigma2 += diff2**2

    ## ToDO recheck whether it's really n or n-1 instead
    # normalize covariance and variance
    covariance /= n 
    sigma1 = sigma1[0] /(n) 
    sigma2 = sigma2[0] /(n) 

    ## init covariance matrix
    cov_matrix = np.array([[sigma1, covariance],[covariance, sigma2]])    

    return cov_matrix


def matrix2dDeterminant(mat: np.ndarray) -> float:
    # EXERCISE 2 - Compute the determinant value of a 2x2 matrix
    matrix_determinant = mat[0][0] * mat[1][1] - (mat[0][1] * mat[1][0])
    return matrix_determinant


def matrix2dInverse(mat: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the inverse matrix of a 2x2 matrix
    det = matrix2dDeterminant(mat)
    a = mat[0][0]

    mat[1][0] = -1*mat[1][0]
    mat[0][1] = -1*mat[0][1] 
    mat[0][0] = mat[1][1]
    mat[1][1] = a
    
    matrix_inverse = 1/det * mat 
    return matrix_inverse


def pdf(x: np.array, mean: np.ndarray, cov: np.ndarray) -> float:
    # EXERCISE 2 - Implement PDF function for a 2D multivariate normal distribution (MVND)
    assert x.shape == mean.shape
    inv = matrix2dInverse(cov)
    det = matrix2dDeterminant(cov)

    mult_one = np.dot(np.transpose(x - mean) , inv)
    mult_two = np.dot( mult_one , x-mean)   

    probability = 1/np.sqrt(4*np.pi**2 * det) * np.exp(-0.5 * mult_two[0][0])

    return probability


def classification(x: np.ndarray, mean_class0: np.ndarray, cov_class0: np.ndarray,
                   mean_class1: np.ndarray, cov_class1: np.ndarray) -> bool:
    # EXERCISE 2 - Implement classification function of a point into one of 2 MVND
    # Todo: Fix dirty tranpose
    x = np.array([[x[0]],[x[1]]])

    # Mahalanobis Distance
    sigma0_inv = matrix2dInverse(cov_class0)
    sigma1_inv = matrix2dInverse(cov_class1)   
    
    mult0_one = np.dot(np.transpose(x - mean_class0) , sigma0_inv)
    mult0_two = np.dot(mult0_one, (x - mean_class0) )
    d0 = mult0_two[0][0]
    
    mult1_one = np.dot(np.transpose(x - mean_class1) , sigma1_inv)
    mult1_two = np.dot( mult1_one , (x - mean_class1) )
    d1 = mult1_two[0][0]

    # Return 1 if class 1 and 0 if class 0.    
    if d0 < d1:
        assigned_class = 0
    else:
        assigned_class = 1   

    return assigned_class


def box_muller_transform(unif1: float, unif2: float) -> (float, float):
    # EXERCISE 2 - Implement sampling from a standard normal distribution
    # Transforms 2 uniform samples into 2 random samples from a standard normal distribution N(0,1)
    rnd1 = np.sqrt(-2 * np.log(unif1)) * np.cos(2*np.pi*unif2)
    rnd2 = np.sqrt(-2 * np.log(unif1)) * np.sin(2*np.pi*unif2)

    return (rnd1, rnd2)


def cholesky_factor_2d(mat: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the cholesky decomposition of a 2x2 matrix
    d, n = np.shape(mat)
    # should only be applied to quadratic 2x2 matrices matrices
    # the matrix should be symmetric
    assert d == n and n == 2
    assert mat[0, 1] - mat[1, 0] < 1e-5
    
    # compare with np function
    #L = np.linalg.cholesky(np.matrix(mat))

    L = np.zeros((n,n))
    ## cholesky deposition
    for i, (Ai, Li) in enumerate(zip(mat, L)):
        for j, Lj in enumerate(L[:i+1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = np.sqrt(Ai[i] - s) if (i == j) else \
                      (1.0 / Lj[j] * (Ai[j] - s))    
    return L


def sample_from_2d_gaussian(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Implement sampling from MVND
    # Sample a random 2D vector from a normal distribution with given mean and covariance matrix
    # Step1: Sample two uniform random values
    random_number_uniform_1 = np.random.random()
    random_number_uniform_2 = np.random.random()
    # Step2: Transform the uniform samples into samples from a standard normal distribution
    # using the box_muller_transform
    (rnd1, rnd2) = box_muller_transform(random_number_uniform_1, random_number_uniform_2)

    # perform cholesky
    L = cholesky_factor_2d(cov)
    z = np.array([[rnd1],[rnd2]])

    sample = np.dot(L, z) + mean

    # Convert the normal samples into a 2D sample from a MVND with the provided mean and covariance matrix
    return sample


def print_classification(data: np.ndarray, labels: np.ndarray, class0_mean: np.ndarray, class0_cov: np.ndarray,
                         class1_mean: np.ndarray, class1_cov: np.ndarray) -> None:
    # Helper function to check if data items are classified correctly
    correctly_classified = 0
    false_class_0 = 0
    false_class_1 = 0

    _, n = np.shape(data)

    for i in range(n):
        if classification(data[:, i], class0_mean, class0_cov, class1_mean, class1_cov):
            if labels[i] == 1:
                correctly_classified += 1
            else:
                false_class_1 += 1

        else:
            if labels[i] == 0:
                correctly_classified += 1
            else:
                false_class_0 += 1

    print('############# DATA CLASSIFICATION #################')
    print('Classified', correctly_classified, 'out of', n, 'samples correctly.')
    print('False omega_1:', false_class_0)
    print('False omega_2:', false_class_1)
    print('###################################################\n')


def plot_cov(ax: plt.axes, mean: np.ndarray, cov_matrix: np.ndarray, color: str) -> None:
    # Helper function to visualize the distribution
    vals, vecs = np.linalg.eigh(cov_matrix)
    print(vals)
    x, y = vecs[:, 0]

    theta = np.degrees(np.arctan2(y, x))
    w, h = 2 * np.sqrt(vals)
    ax.add_artist(Ellipse(mean, w, h, theta, color=color, alpha=0.3))


if __name__ == '__main__':
    data_train = scio.loadmat(os.path.join(dataPath, 'mle_toy_train.mat'))
    data_test = scio.loadmat(os.path.join(dataPath, 'mle_toy_test.mat'))

    class0_training_data = data_train['omega_1']
    
    class0_training_labels = np.zeros((1, np.shape(class0_training_data)[1]))
    class1_training_data = data_train['omega_2']
    class1_training_labels = np.ones((1, np.shape(class1_training_data)[1]))

    class0_test_data = data_test['omega_1']
    class0_test_labels = np.zeros((1, np.shape(class0_test_data)[1]))
    class1_test_data = data_test['omega_2']
    class1_test_labels = np.ones((1, np.shape(class1_test_data)[1]))

    # visualize the data set
    fig1 = plt.figure(1)
    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'r.')
    plt.legend(['$\omega_1$ Training', '$\omega_2$ Training'])
    plt.title('Training Data')
    plt.show(block=False)

    class0_mean = calculate_sample_mean(class0_training_data)
    class1_mean = calculate_sample_mean(class1_training_data)

    class0_cov = calculate_sample_covariance_matrix(class0_training_data, class0_mean)
    class1_cov = calculate_sample_covariance_matrix(class1_training_data, class1_mean)

    print("\nClass0: Estimated mean: {}, \nEstimated covariance matrix: \n{}".format(class0_mean, class0_cov))
    print("\nClass1: Estimated mean: {}, \nEstimated covariance matrix: \n{}".format(class1_mean, class1_cov))

    print("\nTraining data classification:")
    all_training_data = np.append(class0_training_data, class1_training_data, axis=1)
    all_training_labels = np.append(class0_training_labels, class1_training_labels)
    print_classification(all_training_data, all_training_labels, class0_mean, class0_cov, class1_mean, class1_cov)

    print("\nTest data classification:")
    all_test_data = np.append(class0_test_data, class1_test_data, axis=1)
    all_test_labels = np.append(class0_test_labels, class1_test_labels)
    print_classification(all_test_data, all_test_labels, class0_mean, class0_cov, class1_mean, class1_cov)

    class0_random_sample = sample_from_2d_gaussian(class0_mean, class0_cov)
    class1_random_sample = sample_from_2d_gaussian(class1_mean, class1_cov)

    print("\nClass0 Sample: {}, pdf-class0: {}, pdf-class1: {}".format(
        class0_random_sample,
        pdf(class0_random_sample, class0_mean, class0_cov),
        pdf(class0_random_sample, class1_mean, class1_cov))
    )

    print("\nClass1 Sample: {}, pdf-class0: {}, pdf-class1: {}".format(
        class1_random_sample,
        pdf(class1_random_sample, class0_mean, class0_cov),
        pdf(class1_random_sample, class1_mean, class1_cov))
    )

    # visualize estimated normal distributions
    fig2 = plt.figure(2)
    ax = plt.axes()
    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
    plt.plot(class0_test_data[0, :], class0_test_data[1, :], 'bo')
    plt.plot(class0_random_sample[0], class0_random_sample[1], 'k*')
    plot_cov(ax, class0_mean, class0_cov, 'blue')

    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'rx')
    plt.plot(class1_test_data[0, :], class1_test_data[1, :], 'ro')
    plt.plot(class1_random_sample[0], class1_random_sample[1], 'gD')
    plot_cov(ax, class1_mean, class1_cov, 'red')
    plt.legend(['$\omega_0$ Training', '$\omega_0$ Test', '$\omega_0$ Sample',
                '$\omega_1$ Training', '$\omega_1$ Test', '$\omega_1$ Sample'])
    plt.title('Training and test Data')
    plt.show()