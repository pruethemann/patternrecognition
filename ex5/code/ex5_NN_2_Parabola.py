import sys
from trainer import Trainer
import numpy as np
import scipy.io as scio
from mySimpleNN import mySimpleNN
import torch


def parabolaData() -> None:
    X_train = np.load('../data/train_inputs.npy').T.astype(np.float32)
    y_train = np.load('../data/train_targets.npy').astype(np.float32)
    X_test = np.load('../data/validation_inputs.npy').T.astype(np.float32)
    y_test = np.load('../data/validation_targets.npy').astype(np.float32)

    nTrainSamples = X_train.shape[0]
    print("Total number of training examples: {}".format(nTrainSamples))


    # TODO: Define model, optimizer and loss
    # Define training batch size
    batch_size = 50
    # Initialize the 'mySimpleNN' instance
    model = mySimpleNN((batch_size, 2))
    # Define optimizer to use from torch.optim.xxx
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Define model cost function torch.nn.xxx
    criterion = torch.nn.MSELoss(reduction='sum')
    # Total number of epocs to execute
    epocs = 500

    trainer = Trainer(model, optimizer, criterion)
    trainer.trainModel(X_train, y_train, X_test, y_test,
                       num_of_epochs_total=epocs, batch_size=batch_size, output_folder='output/parabola/')


def flowerData() -> None:
    train = scio.loadmat('../data/flower.mat')['train']
    d, n = np.shape(train)
    y_train = np.reshape(train[-1, :], (1, n)).astype(np.float32)
    X_train = train[:-1, :].T.astype(np.float32)

    test = scio.loadmat('../data/flower.mat')['test']
    d, n = np.shape(test)
    y_test = np.reshape(test[-1, :], (1, n)).astype(np.float32)
    X_test = test[:-1, :].T.astype(np.float32)

    nTrainSamples = X_train.shape[0]
    print("Total number of training examples: {}".format(nTrainSamples))


    # TODO: Define model, optimizer and loss
    # Define training batch size
    batch_size = 30
    # Initialize the 'mySimpleNN' instance
    model = mySimpleNN((batch_size, 2))
    # Define optimizer to use from torch.optim.xxx
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Define model cost function torch.nn.xxx
    criterion = torch.nn.MSELoss(reduction='sum')
    # Total number of epocs to execute
    epocs = 500

    trainer = Trainer(model, optimizer, criterion)
    trainer.trainModel(X_train, y_train, X_test, y_test,
                       num_of_epochs_total=epocs, batch_size=batch_size, output_folder='output/flower/')


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network parabola example!")
    parabolaData()
    print("##########-##########-##########")
    print("##########-##########-##########")
    print("##########-##########-##########")
    print("Neural network flower example!")
    # flowerData()
    print("##########-##########-##########")
