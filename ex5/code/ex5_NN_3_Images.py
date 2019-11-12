import numpy as np
import time, os

import torch
from torchvision import datasets
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from myImageNN import MyCNN, MyFullyConnectedNN, MyLogRegNN


def writeHistoryPlots(history, modelType, filePath):
    history = np.array(history)
    plt.clf()
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_loss_curve.png')
    plt.clf()
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_accuracy_curve.png')



def LoadDataSet() -> (int, DataLoader, int, DataLoader):
    dataset = '../data/horse_no_horse/'
    # TODO: Load image dataset
    # Hint: see the Transer_learning notebook on how this can be done

    # mean, std_dev = mean_and_standard_dev(train_data_loader)
    # print(f"Training data mean: {mean}, std-dev: {std_dev}")

    train_data_loader = DataLoader(???)
    valid_data_loader = DataLoader(???)

    return train_data_size, train_data_loader, valid_data_size, valid_data_loader


def train_and_validate(myModel, criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param myModel: Model to train and validate
        :param criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    train_data_size, train_data_loader, valid_data_size, valid_data_loader = LoadDataSet()
    history = []

    # TODO: Train model and validate model on validation set after each epoch
    # Hint: see the Transer_learning notebook and the trainer class on how this can be done
    return myModel, history


if __name__ == '__main__':
    # # TODO: train and test a logistic regression classifier implemented as a neural network
    print('##########################')
    print('Testing Logistic Regression')
    logRegModel = MyLogRegNN()

    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    finallogRegmodel, logRegHistory = train_and_validate(logRegModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(logRegHistory, 'logRegModel', 'output/')

    # TODO: train and test the fully connected DNN
    print('##########################')
    print('Testing Deep Neural Net')
    dnnModel = MyFullyConnectedNN()
    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    finalDNNmodel, dnnHistory = train_and_validate(dnnModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(dnnHistory, 'dnnModel', 'output/')

    # TODO: train and test a CNN
    print('##########################')
    print('Testing Convolutional Neural Net')
    cnnModel = MyCNN()
    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function

    finalCNNmodel, cnnHistory = train_and_validate(cnnModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(cnnHistory, 'cnnModel', 'output/')
