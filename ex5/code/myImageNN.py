import torch


class MyLogRegNN(torch.nn.Module):

    def __init__(self):
        super(MyLogRegNN, self).__init__()
        # TODO: Define a logistic regression classifier as a neural network
        self.linear = torch.nn.Linear(3072, 2)

    def forward(self, x):
        # transform = x.view(-1, 3072)
        # hidden = self.linear(transform)
        transform = x.view(-1, 3072)
        y_hat = self.linear(transform)
        return y_hat


class MyFullyConnectedNN(torch.nn.Module):
    def __init__(self):
        super(MyFullyConnectedNN, self).__init__()
        # TODO: Define a fully connected neural network
        # Using DNN from mySimpleNN with fewer nodes
        dimHidden = 10
        self.layer_1 = torch.nn.Linear(3072, dimHidden)
        self.layer_2 = torch.nn.Linear(dimHidden, dimHidden)
        self.layer_3 = torch.nn.Linear(dimHidden, 2)

    def forward(self, x):
        transform = x.view(-1, 3072)
        h1 = self.layer_1(transform).clamp(min=0)
        h2 = self.layer_2(h1).clamp(min=0)
        y_hat = self.layer_3(h2)
        return y_hat


class MyCNN(torch.nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # TODO: Define a convolutional neural network

    def forward(self, x):
        y_hat = None
        return y_hat
