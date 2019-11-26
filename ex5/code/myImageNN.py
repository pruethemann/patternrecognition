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
        self.dimImages = 32 * 32 * 3
        dimHidden = 10
        classes = 2
        self.layer_1 = torch.nn.Linear(self.dimImages, dimHidden)
        self.layer_2 = torch.nn.Linear(dimHidden, dimHidden)
        self.layer_3 = torch.nn.Linear(dimHidden, classes)

    def forward(self, x):
        transform = x.view(-1, self.dimImages)
        h1 = self.layer_1(transform).clamp(min=0)
        h2 = self.layer_2(h1).clamp(min=0)
        y_hat = self.layer_3(h2)
        return y_hat


class MyCNN(torch.nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # TODO: Define a convolutional neural network
        # https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch
        self.dimImages = 32 * 32 * 3
        classes = 2
        dimHidden = 10

        # Convolution
        self.convolution = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        # Max Pooling
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Connected Layers
        self.hidden_layer1 = torch.nn.Linear(18 * 16 * 16, dimHidden)
        self.hidden_layer2 = torch.nn.Linear(dimHidden, classes)

    def forward(self, x):
        # conv = torch.nn.ReLU(self.convolution(x))
        # pool = self.pooling(conv)
        # h1 = pool.view(-1, 18 * 16 * 16)
        # h2 = torch.nn.ReLU(self.hidden_layer1(h1))
        # y_hat = self.hidden_layer2(h2)
        conv = self.convolution(x)
        pool = self.pooling(conv)
        view = pool.view(-1, 18 * 16 * 16)
        h1 = self.hidden_layer1(view)
        y_hat = self.hidden_layer2(h1)
        return y_hat
