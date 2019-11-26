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
    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=transforms.ToTensor()),
        'valid': datasets.ImageFolder(root=valid_directory, transform=transforms.ToTensor())
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # mean, std_dev = mean_and_standard_dev(train_data_loader)
    # print(f"Training data mean: {mean}, std-dev: {std_dev}")

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], shuffle=True)
    valid_data_loader = DataLoader(data['valid'], shuffle=True)

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

    # TODO: Train model and validate model on validation set after each epoch
    # Hint: see the Transer_learning notebook and the trainer class on how this can be done

    start = time.time()
    history = []
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        myModel.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = myModel(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            myModel.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = myModel(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))

        # Save if the model has best accuracy till now
        # torch.save(model, dataset+'_model_'+str(epoch)+'.pt')

    return myModel, history


if __name__ == '__main__':
    # TODO: train and test a logistic regression classifier implemented as a neural network
    # print('##########################')
    # print('Testing Logistic Regression')
    # logRegModel = MyLogRegNN()
    #
    # criterion = torch.nn.CrossEntropyLoss()  # Cost function - torch.nn.XXX loss functions
    # optimizer = torch.optim.SGD(logRegModel.parameters(), lr=1e-3)  # Optimizer algorithm - torch.optim.XXX function
    # finallogRegmodel, logRegHistory = train_and_validate(logRegModel, criterion, optimizer, epochs=20)
    # writeHistoryPlots(logRegHistory, 'logRegModel', 'output/')

    # TODO: train and test the fully connected DNN
    print('##########################')
    print('Testing Deep Neural Net')
    dnnModel = MyFullyConnectedNN()
    criterion = torch.nn.CrossEntropyLoss()  # Cost function - torch.nn.XXX loss functions
    optimizer = torch.optim.Adam(dnnModel.parameters(), lr=1e-4)  # Optimizer algorithm - torch.optim.XXX function
    finalDNNmodel, dnnHistory = train_and_validate(dnnModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(dnnHistory, 'dnnModel', 'output/')

    # TODO: train and test a CNN
    # print('##########################')
    # print('Testing Convolutional Neural Net')
    # cnnModel = MyCNN()
    # criterion = None  # Cost function - torch.nn.XXX loss functions
    # optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    #
    # finalCNNmodel, cnnHistory = train_and_validate(cnnModel, criterion, optimizer, epochs=20)
    # writeHistoryPlots(cnnHistory, 'cnnModel', 'output/')
