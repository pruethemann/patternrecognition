# implements the ada_boost class for an ada boost learner
import numpy as np


class AdaBoost:
    # two np vectors
    sample_weights_1: []
    sample_weights_2: []
    # two arrays of numpy matrices
    data_1: []
    data_2: []
    # two arrays of sample labels
    labels_1: []
    labels_2: []
    # an array of decision stumps
    stumps: []
    # an array of consecutively learnt hypothesis weights
    hypothesis_weights: []
    T: int

    def __init__(self, input_data_1, input_data_2, input_T):
        n_1, d = np.shape(input_data_1)
        self.data_1 = input_data_1[:, :d-1]
        self.labels_1 = input_data_1[:, d-1]

        n_2, d = np.shape(input_data_2)
        self.data_2 = input_data_2[:, :d-1]
        self.labels_2 = input_data_2[:, d-1]

        self.sample_weights_1 = np.array([1/(n_1 + n_2) for i in range(n_1)])
        self.sample_weights_2 = np.array([1/(n_1 + n_2) for i in range(n_2)])
        self.T = input_T
        
        self.stumps = []
        self.hypothesis_weights = []


    def train(self):
        n_1, d = np.shape(self.data_1)
        n_2, d = np.shape(self.data_2)

        data = np.append(self.data_1, self.data_2, axis=0)

        for s in range(self.T):
            for t in range(d):
                # creating a decision stump
                k = t % d
                k_min = np.min(data[:, k])
                k_max = np.max(data[:, k])

                k_stump = k_min + (k_max - k_min)*np.random.random()

                # a-priori, we assume that class 1 is positive.
                # append the hypotheses
                h_1s = []
                h_2s = []

                eps_1 = 0
                for j in range(n_1):
                    if self.data_1[j, k] >= k_stump:
                        h_1s.append(1)
                    else:
                        h_1s.append(-1)
                        eps_1 += self.sample_weights_1[j]

                eps_2 = 0
                for j in range(n_2):
                    if self.data_2[j, k] >= k_stump:
                        h_2s.append(1)
                        eps_2 += self.sample_weights_2[j]
                    else:
                        h_2s.append(-1)

                # eps = (eps_1*n_1 + eps_2*n_2)/(n_1 + n_2)
                eps = eps_1 + eps_2
                self.stumps.append((k, k_stump))

                # get the hypothesis weight
                # avoid problems with 0
                if eps > 1-1e-10:
                    epsilon = 1-1e-10
                elif eps < 1e-10:
                    epsilon = 1e-10
                else:
                    epsilon = eps

                a = 1/2*np.log((1-epsilon)/epsilon)
                self.hypothesis_weights.append(a)

                h_1s = np.array(h_1s)
                h_2s = np.array(h_2s)

                factor = np.exp(-a*h_1s)
                self.sample_weights_1 = np.multiply(self.sample_weights_1, factor)

                factor = np.exp(a*h_2s)
                self.sample_weights_2 = np.multiply(self.sample_weights_2, factor)

                Z = np.sum(self.sample_weights_1)
                Z += np.sum(self.sample_weights_2)
                
                self.sample_weights_1 /= Z
                self.sample_weights_2 /= Z

    def classify(self, element):
        f = 0
        for i in range(len(self.stumps)):
            k, k_stump = self.stumps[i]
            if element[k] >= k_stump:
                f += self.hypothesis_weights[i]
            else:
                f -= self.hypothesis_weights[i]
        if f >= 0:
            return 1
        else:
            return -1

