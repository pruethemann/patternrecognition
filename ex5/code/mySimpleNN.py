import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self, input_shape: tuple):
        super(mySimpleNN, self).__init__()
        # TODO: Define a simple neural network
        dimIn, dimOut = input_shape
        dimHidden = 200
        self.layer_1 = torch.nn.Linear(dimOut, dimHidden)
        self.layer_2 = torch.nn.Linear(dimHidden, dimHidden)
        self.layer_3 = torch.nn.Linear(dimHidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define the network forward propagation from x -> y_hat
        # h1 = self.layer_1(x).sigmoid()
        h1 = self.layer_1(x).clamp(min=0)
        # h2 = self.layer_2(h1).sigmoid()
        h2 = self.layer_2(h1).clamp(min=0)
        y_hat = self.layer_3(h2)

        return y_hat

