import numpy as np
import matplotlib.pyplot as plt


def plot2D(ax: plt, X: np.ndarray, y: np.ndarray, w: np.ndarray, name: str) -> None:
    '''
    Visualize decision boundary and data classes in 2D
    :param ax: matplotlib
    :param X: data
    :param y: data labels
    :param w: model parameters
    :param name:
    :return:
    '''
    x1 = np.array(X[1, :])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2, :])
    posterior1 = LOGREG().activationFunction(w, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o', '+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        ax.scatter(x1_sub, x2_sub, c=posterior1_sub, vmin=0, vmax=1, marker=m_sub,
                   label='ground truth label = ' + str(li))
    cbar = ax.colorbar()
    cbar.set_label('posterior value')
    ax.legend()
    x = np.arange(x1.min(), x1.max(), 0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p, m) in pms:
        yp = (- np.log((1 / p) - 1) - w[1] * x - w[0]) / w[2]
        yp = np.squeeze(np.asarray(yp))
        ax.plot(x, yp, m, label='p = ' + str(p))
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


def plot3D(ax: plt, sub3d: plt, X: np.ndarray, y: np.ndarray, w: np.ndarray, name: str) -> None:
    '''
    Visualize decision boundary and data classes in 3D
    :param ax:  matplotlib
    :param sub3d: fig.add_subplot(XXX, projection='3d')
    :param X: data
    :param y: data labels
    :param w: model parameters
    :param name: plot name identifier
    :return:
    '''
    x1 = np.array(X[1, :])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2, :])
    posterior1 = LOGREG().activationFunction(w, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o', '+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        sub3d.scatter(x1_sub, x2_sub, posterior1_sub, c=posterior1_sub, vmin=0, vmax=1, marker=m_sub,
                      label='ground truth label = ' + str(li))
    ax.legend()
    x = np.arange(x1.min(), x1.max(), 0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p, m) in pms:
        yp = (- np.log((1 / p) - 1) - w[1] * x - w[0]) / w[2]
        yp = np.squeeze(np.asarray(yp))
        z = np.ones(yp.shape) * p
        sub3d.plot(x, yp, z, m, label='p = ' + str(p))
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


class LOGREG(object):
    '''
    Logistic regression class based on the LOGREG lecture slides
    '''

    def __init__(self, regularization: float = 0):
        self.r = regularization
        self._threshold = 1e-10

    def activationFunction(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        # TODO: Implement logistic function

        # The Logistic Regression Posterior (page 19)
        return 1 / (1 + np.exp(-(np.dot(w.transpose(), X))))

    def _costFunction(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the cost function for the current model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: cost
        '''
        # TODO: Implement equation of cost function for posterior p(y=1|X,w)

        # Maximum Likelihood Estimate of w (page 26)
        cost = 0
        regularizationTerm = self.r

        for i in range(X.shape[1]):
            # cost += y[i] * np.dot(w.transpose(), X[:, i]) - np.log(1 + np.exp(np.dot(w.transpose(), X[:, i])))
            # factor = y[i] * w.transpose()
            cost += np.dot(y[i] * w.transpose(), X[:, i]) - np.log(1 + np.exp(np.dot(w.transpose(), X[:, i])))

        return cost + regularizationTerm

    def _calculateDerivative(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of the model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: first derivative of the model parameters
        '''
        # TODO: Calculate derivative of loglikelihood function for posterior p(y=1|X,w)

        # Maximum Likelihood Estimate of w (page 29). Derivative shape = (3, 1)
        regularizationTerm = self.r

        # Try No. 1
        # firstDerivative = np.zeros((1, 3))
        # for i in range(len(y)):
        #     factor1 = (y[i] - self.activationFunction(w, X[:, i])).reshape((1, 1))
        #     factor2 = X[:, i].reshape((3, 1)).transpose()
        #     firstDerivative += np.dot(factor1, factor2)
        #
        # return firstDerivative.transpose()

        # Try No. 2
        y = y.reshape(y.shape[0], 1)
        firstDerivative = np.dot(X, (np.reshape(self.activationFunction(w, X), (X.shape[1], 1)) - y.reshape((-1, 1))))
        # regularizationTerm = np.reshape(self.r * 2 * np.transpose(w), (X.shape[0], 1))

        return firstDerivative + regularizationTerm

    def _calculateHessian(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        '''
        :param w: current model parameters
        :param X: data
        :return: the hessian matrix (second derivative of the model parameters)
        '''
        # TODO: Calculate Hessian matrix of loglikelihood function for posterior p(y=1|X,w)

        # ======================================================================================================
        # Hessian: Concave Likelihood (page 32)
        regularizationTerm = self.r
        temp = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            xiReshape = X[:, i].reshape((3, 1))
            factor1 = self.activationFunction(w, xiReshape)
            factor2 = 1 - self.activationFunction(w, xiReshape)
            temp[i][i] = factor1 * factor2
        hessian = np.dot(X, temp)
        hessian = np.dot(hessian, X.transpose())

        return hessian + regularizationTerm

    def _optimizeNewtonRaphson(self, X: np.ndarray, y: np.ndarray, number_of_iterations: int) -> np.ndarray:
        '''
        Newton Raphson method to iteratively find the optimal model parameters (w)
        :param X: data
        :param y: data labels (0 or 1)
        :param number_of_iterations: number of iterations to take
        :return: model parameters (w)
        '''
        # TODO: Implement Iterative Re-weighted Least Squares algorithm for optimization, use the calculateDerivative and calculateHessian functions you have already defined above
        w = np.zeros((X.shape[0], 1))  # Initializing the w vector as a numpy matrix class instance

        # Iterative Re-weighted Least Squares (page 33)
        posteriorloglikelihood = self._costFunction(w, X, y)
        print('initial posteriorloglikelihood', posteriorloglikelihood, 'initial likelihood',
              np.exp(posteriorloglikelihood))

        for i in range(number_of_iterations):
            # ======================================================================================================
            oldposteriorloglikelihood = posteriorloglikelihood
            w_old = w
            h = self._calculateHessian(w, X)

            w_update = np.dot(np.linalg.inv(h), self._calculateDerivative(w_old, X, y))
            w = w_old - w_update

            posteriorloglikelihood = self._costFunction(w, X, y)
            if self.r == 0:
                # TODO: What happens if this condition is removed?
                # Cannot test with regularization = 0
                if np.exp(posteriorloglikelihood) > 0.99:
                    print('posterior > 0.99, breaking optimization at niter = ', i)
                    break

            if self.r > 0:
                # TODO: What happens if this condition is removed?
                #
                if np.exp(posteriorloglikelihood) - np.exp(oldposteriorloglikelihood) < 0:  # posterior is decreasing
                    print('negative loglikelihood increasing, breaking optimization at niter = ', i)
                    posteriorloglikelihood = oldposteriorloglikelihood
                    w = w_old
                    break
            # TODO: Implement convergenc check based on when w_update is close to zero
            # Note: You can make use of the class threshold value self._threshold
            if np.sum(np.abs(w_update)):
                break

        print('final posteriorloglikelihood', posteriorloglikelihood, 'final likelihood',
              np.exp(posteriorloglikelihood))

        # Note: maximize likelihood (should become larger and closer to 1), maximize loglikelihood( should get less negative and closer to zero)
        return w

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int) -> np.ndarray:
        '''
        :param X: dataset
        :param y: ground truth labels
        :param iterations: Number of iterations to train
        :return: trained w parameter
        '''
        self.w = self._optimizeNewtonRaphson(X, y, iterations)
        return self.w

    def classify(self, X: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained logistic regressor - access the w parameter through self.
        :param x: Data to be classified
        :return: List of classification values (0.0 or 1.0)
        '''
        # TODO: Implement classification function for each entry in the data matrix
        numberOfSamples = X.shape[1]
        predictions = self.activationFunction(self.w, X)
        # Rounding predictions to 0.0 respectively 1.0
        return np.around(predictions)

    def printClassification(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls "classify" and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement print classification
        numberOfSamples = X.shape[1]
        predictions = self.classify(X)

        # if prediction - y != 0 => misclassified!
        numOfMissclassified = np.sum(np.abs(predictions - y))
        totalError = 100.0 / numberOfSamples * numOfMissclassified

        print("{}/{} misclassified. Total error: {:.2f}%.".format(numOfMissclassified, numberOfSamples, totalError))
