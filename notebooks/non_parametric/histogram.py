# This class implements the Histogram
import numpy as np
import matplotlib.pyplot as plt


class Bin:
    lower_bounds: []
    upper_bounds: []
    volume: float
    counter: int

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        vol = 1
        for i in range(len(self.lower_bounds)):
            a = self.lower_bounds[i]
            b = self.upper_bounds[i]
            vol = vol * (b - a)
        self.volume = vol
        self.counter = 0

    # might be unnecessary...
    def __cmp__(self, other):
        for i in range(len(self.lower_bounds)):
            if self.lower_bounds[i] != other.lower_bounds[i]:
                return self.lower_bounds[i].__cmp__other.lower_bounds[i]
        return 0

    def get_lower_bounds(self):
        return self.lower_bounds

    def get_upper_bounds(self):
        return self.upper_bounds


class Histogram:
    d: int
    n_o_points: int
    boundaries: []
    min_values: []
    max_values: []
    bins: {}

    def __init__(self, data, number_of_bins):
        n, d = np.shape(data)
        self.d = len(number_of_bins)
        self.n_o_points = np.shape(data)[0]
        self.boundaries = []
        self.min_values = []
        self.max_values = []
        self.bins = {}

        # create bins
        for i in range(self.d):
            data_min = float(np.min(data[:, i]))
            data_max = float(np.max(data[:, i]))
            self.min_values.append(data_min)
            self.max_values.append(data_max)

            data_range = data_max - data_min
            current_boundaries = [data_min + j*data_range/number_of_bins[i] for j in range(1, number_of_bins[i]-1)]
            self.boundaries.append(current_boundaries)

            if i == 0:
                self.bins[((data_min,), (current_boundaries[0],))] = Bin([data_min], [current_boundaries[0]])
                self.bins[((current_boundaries[-1],), (data_max,))] = Bin([current_boundaries[-1]], [data_max])
                for j in range(1, len(current_boundaries)):
                    key = ((current_boundaries[j-1],), (current_boundaries[j],))
                    self.bins[key] = Bin(list(key[0]), list(key[1]))
            else:
                new_bins = {}
                for key in self.bins:
                    bin_lower = list(key[0]).copy()
                    bin_upper = list(key[1]).copy()
                    bin_lower.append(data_min)
                    bin_lower_tuple = tuple(bin_lower)
                    bin_upper.append(current_boundaries[0])
                    bin_upper_tuple = tuple(bin_upper)

                    new_bins[(bin_lower_tuple, bin_upper_tuple)] = Bin(bin_lower, bin_upper)

                    bin_lower = list(key[0]).copy()
                    bin_upper = list(key[1]).copy()
                    bin_lower.append(current_boundaries[-1])
                    bin_lower_tuple = tuple(bin_lower)
                    bin_upper.append(data_max)
                    bin_upper_tuple = tuple(bin_upper)
                    new_bins[(bin_lower_tuple, bin_upper_tuple)] = Bin(bin_lower, bin_upper)

                    for j in range(1, len(current_boundaries)):
                        bin_lower = list(key[0]).copy()
                        bin_upper = list(key[1]).copy()
                        bin_lower.append(current_boundaries[j-1])
                        bin_lower_tuple = tuple(bin_lower)
                        bin_upper.append(current_boundaries[j])
                        bin_upper_tuple = tuple(bin_upper)
                        new_bins[(bin_lower_tuple, bin_upper_tuple)] = Bin(bin_lower, bin_upper)

                self.bins = new_bins

        # train the algorithm
        for i in range(n):
            # iterate over all bins
            bucket = self.get_appropriate_bin(data[i, :])
            bucket.counter += 1

    def get_appropriate_bin(self, x):
        # use the dictionary structure of self.bins
        # get the boundaries
        lower_bounds = []
        upper_bounds = []
        for i in range(self.d):
            if x[i] < self.min_values[i] or x[i] > self.max_values[i]:
                return None

            current_bounds = self.boundaries[i]
            last = True
            for j in range(len(current_bounds)):
                if x[i] <= current_bounds[j]:
                    last = False
                    if j == 0:
                        lower_bounds.append(self.min_values[i])
                    else:
                        lower_bounds.append(current_bounds[j-1])
                    upper_bounds.append(current_bounds[j])
                    break
            if last:
                lower_bounds.append(current_bounds[-1])
                upper_bounds.append(self.max_values[i])

        lower_bounds = tuple(lower_bounds)
        upper_bounds = tuple(upper_bounds)

        bucket = self.bins[(lower_bounds, upper_bounds)]
        return bucket

    def p(self, x):
        bucket = self.get_appropriate_bin(x)

        if bucket is None:
            return 0

        return (bucket.counter / self.n_o_points) / bucket.volume


if __name__ == '__main__':
    length = 100000
    data = np.reshape(np.random.normal(0, 1, size=length), (length, 1))
    hist = Histogram(data, [100])

    bin = hist.get_appropriate_bin(np.reshape(np.array(0), (1, 1)))

    x = np.linspace(-5, 5, 500)
    x = np.reshape(x, (len(x), 1))
    y = np.zeros(np.shape(x))

    for i in range(len(x)):
        y[i] = hist.p(x[i, :])

    plt.plot(x, y)
    plt.show()

    mean = (0, 0)
    cov = np.eye(2)
    data_2 = np.reshape(np.random.multivariate_normal(mean, cov, size=length), (length, 2))
    hist_2 = Histogram(data_2, [100, 100])

    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)

    xm, ym = np.meshgrid(x, y)
    n, m = np.shape(xm)

    z = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            z[i, j] = hist_2.p(np.array([xm[i, j], ym[i, j]]))

    plt.contourf(x, y, z)
    plt.show()
