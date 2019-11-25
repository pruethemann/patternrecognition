import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self, input_shape: tuple):
        super(mySimpleNN, self).__init__()
        # TODO: Define a simple neural network
        dimIn, dimOut = tuple
        dimHidden = 30
        self.hidden_1 = torch.nn.Linear(dimIn, dimHidden)
        self.hidden_2 = torch.nn.Linear(dimHidden, dimHidden)
        self.hidden_3 = torch.nn.Linear(dimHidden, dimOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define the network forward propagation from x -> y_hat
        h1 = self.hidden_1(x).sigmoid()
        h2 = self.hidden_2(h1).sigmoid()
        y_hat = self.hidden_3(h2)

        return y_hat

