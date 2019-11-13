import numpy as np
import math
import csv
from random import randint

class Node:
    isLeaf = False
    s = np.zeros((0, 0))
    children = []
    MIN_REMAINING_VALUES: int
    separations: []
    omega = None
    a_star: int
    s_nus: []

    def __init__(self, s, min_rem_values, separations):
        self.s = s
        self.MIN_REMAINING_VALUES = min_rem_values
        self.separations = separations

    def entropy(self, S_nu):
        n, d = np.shape(S_nu)
        if n == 0:
            return 0
        p = 0
        # compute the prior
        for i in range(n):
            if S_nu[i, d-1] == 0:
                p += 1

        p = p/n

        if p == 0 or p == 1:
            return 0
        else:
            return -p*math.log(p, 2) - (1-p)*math.log(1-p, 2)

    # returns IG and the set of S_nus
    def ig(self, a):
        S_nus = self.get_s_nus(a)
        h_nu = 0
        n, d = np.shape(self.s)

        for S_nu in S_nus:
            if S_nu is None:
                continue
            m, e = np.shape(S_nu)
            h_nu += (m/n)*self.entropy(S_nu)

        return self.entropy(self.s) - h_nu, S_nus

    def get_s_nus(self, a):
        thresholds = self.separations[a]
        if len(thresholds) == 0:
            return [self.s]

        s = self.s.copy()

        S_nus = []

        for k in range(len(thresholds)+1):
            n, d = np.shape(s)
            if k == len(thresholds):
                S_nus.append(s)
            else:
                deleted = []
                S_k = np.zeros((0, d))
                appended_something = False
                for i in range(n):
                    if s[i, a] <= thresholds[k]:
                        S_k = np.append(S_k, np.reshape(s[i, :], (1, d)), axis=0)
                        appended_something = True
                        deleted.append(i)

                if appended_something:
                    S_nus.append(S_k)
                    for j in reversed(deleted):
                        s = np.delete(s, j, 0)
                else:
                    S_nus.append(None)

        return S_nus

    def get_a_star(self):
        ig_max = - float('inf')
        a_max = None
        s_nus_max = None
        n, d = np.shape(self.s)
        for a in range(d-1):
            ig, s_nus = self.ig(a)
            if ig > ig_max:
                ig_max = ig
                self.a_star = a
                self.s_nus = s_nus

        return a_max, s_nus_max

    def train(self):
        n, d = np.shape(self.s)

        # check whether all labels are equal
        uniformClass = True
        for i in range(n):
            if self.s[i, d-1] != self.s[0, d-1]:
                uniformClass = False
                break

        if uniformClass:
            self.isLeaf = True
            self.omega = self.s[0, d-1]
            return
        # check whether there are too few samples or no more features to consider
        elif n <= self.MIN_REMAINING_VALUES or d == 1:
            self.isLeaf = True
            omega_zeros = 0
            for i in range(n):
                if self.s[i, d-1] == 0:
                    omega_zeros += 1

            if omega_zeros >= n - omega_zeros:
                self.omega = 0
            else:
                self.omega = 1
            return

        # else apply the tree policy recursively

        self.get_a_star()

        children = []
        for s_nu in self.s_nus:
            if s_nu is None:
                children.append(None)
                continue
            m, e = np.shape(s_nu)
            if m == 0:
                children.append(None)
                continue
            s_nu = np.delete(s_nu, self.a_star, 1)
            separations_children = self.separations.copy()
            del separations_children[self.a_star]
            node = Node(s_nu, self.MIN_REMAINING_VALUES, separations_children)
            node.train()
            children.append(node)
        self.children = children

    def classify(self, element):
        if self.isLeaf:
            return self.omega
        else:
            # handle unseen data here!!
            if element[self.a_star] is None:
                n, d = np.shape(self.s)
                mean = 0
                for i in range(n):
                    mean += self.s[i, self.a_star]
                mean /= n
                element[self.a_star] = mean
            thresholds = self.separations[self.a_star]
            for k in range(len(thresholds)):
                if k == len(self.children):
                    break
                elif element[self.a_star] <= thresholds[k]:
                    if self.children[k] is not None:
                        return self.children[k].classify(np.delete(element, self.a_star))
                    else:
                        continue
            # iterated over the while array: take the last entry that is not none
            k = len(self.children) - 1
            while self.children[k] is None:
                k -= 1
            return self.children[k].classify(np.delete(element, self.a_star))

    def to_string(self, features, depth):
        if self.isLeaf:
            return ''
        string = str(features[self.a_star])
        if len(self.children) > 0:
            features = np.delete(features, self.a_star)
            for child in self.children:
                if child is not None and not child.isLeaf:
                    string += '\n'
                    for i in range(depth):
                        string += '|  '
                    string += '|--'
                    string += child.to_string(features, depth + 1)
        
        return string
		

def formatize(data_prime):
    data = data_prime.copy()

    # remove the names
    data = np.delete(data, 1, 1)

    # convert the sex to a numerical value: 0 for male, 1 for female
    n, d = np.shape(data)
    for i in range(n):
        if data[i, 1] == 'male':
            data[i, 1] = 0
        else:
            data[i, 1] = 1

    # remove the categories description
    data = data[1:, :]
    # take numerical values
    data = data.astype(float)

    return data

def split_data(data, test_set_size):
    training_data = data.copy()
    n, d = np.shape(training_data)
    test_data = np.zeros((0, d))
    for i in range(test_set_size):
        k = randint(0, n-i-1)
        test_data = np.append(test_data, np.reshape(training_data[k, :], (1, d)), axis=0)
        training_data = np.delete(training_data, k, 0)

    return training_data, test_data
