import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils
from skimage import io, transform
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import pickle

# Train and test dataset classes for loading the data to the PyTorch model easily

class TrainTrafficDataset(Dataset):
    def __init__(self, datapath, transform):
        self.img_paths = glob.glob(os.path.join(datapath, "*", "*.png"))
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)/255
        if self.transform:
            image = self.transform(image)
        #breakpoint()
        class_id = int(img_path.split('/')[2])
        return image, class_id

class TestTrafficDataset(Dataset):
    def __init__(self, datapath, transform):
        self.img_paths = glob.glob(os.path.join(datapath, "*.png"))
        self.transform = transform
    def __len__(self):
            return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)/255
        if self.transform:
            image = self.transform(image)
        class_id = int(img_path.split('/')[2].split('_')[0])
        return image, class_id

# Function that evaluates a model accuracy in train and test set

def evaluate(model, trainloader, testloader, device):
    correct_train, total_train, correct_test, total_test = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    train_accuracy = correct_train * 100 / total_train
    test_accuracy = correct_test * 100 / total_test
    return train_accuracy, test_accuracy

# Function associated to training a model for a given number of epochs and evaluating the model performance during
# each epoch

def train(model, criterion, optimizer, epochs, trainloader, testloader, device, savepath=None):

    # Initializes lists that will record progress throughout training
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Initializes the best accuracies in test set during this training and among all the previously trained models
    best_acc = 0
    try:
        with open('best_test_accuracy.pkl', 'rb') as file:
            best_global_acc = pickle.load(file)
    except:
        best_global_acc = 0

    # Training during the indicated number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Evaluating")
        train_accuracy, test_accuracy = evaluate(model, trainloader, testloader, device)
        print(f"Epoch {epoch} - Loss: {running_loss} train_acc: {train_accuracy} test_acc: {test_accuracy}")
        loss_list.append(running_loss)
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

    # Saves the best performance of this training with the model and experiment id
        if savepath is not None and test_acc_list[-1] > best_acc:
            best_acc = test_acc_list[-1]
            torch.save(model.state_dict(), savepath)
    # If this model achieves the overall best performance among models it is saved as "best_model.pth"
        if best_acc > best_global_acc:
            best_global_acc = best_acc
            with open('best_test_accuracy.pkl', 'wb') as file:
                pickle.dump(best_global_acc, file)
            torch.save(model.state_dict(), os.path.join("models", "best_model.pth"))

    return model, loss_list, train_acc_list, test_acc_list

# Function that creates and saves loss and accuracy graphs for training and test sets

def result_visualization(epochs, loss_list, train_acc_list, test_acc_list, model_id, experiment_id, save=False):

    plt.figure()
    plt.plot(range(epochs), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Model {model_id} Experiment {experiment_id}")
    plt.grid()
    if save:
        plt.savefig(os.path.join("results", "loss_"+str(model_id)+"_"+str(experiment_id)+".png"))
    plt.show()

    plt.figure()
    plt.plot(range(epochs), train_acc_list, "r", label="Train Accuracy")
    plt.plot(range(epochs), test_acc_list, "b", label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - Model {model_id} Experiment {experiment_id}")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(os.path.join("results", "acc_" + str(model_id) + "_" + str(experiment_id) + ".png"))
    plt.show()

# Main Function

def main(epochs, lr, batch_size, model_id, experiment_id, transfer):

    # Statement that allows employing GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datapath = os.path.join("traffic_Data", "DATA")
    test_datapath = os.path.join("traffic_Data", "TEST")

    # This transformations fix the model input size of the image to 128x128, and normalizes its intensity according to
    # the training set distribution. It also applies data augmentation
    tsfm_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.ColorJitter(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective()

    ])
    # The test set dataloader does not apply any data augmentation transformation
    tsfm_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    trainset = TrainTrafficDataset(train_datapath, tsfm_train)
    testset = TestTrafficDataset(test_datapath, tsfm_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # This first model parts form a pretrained AlexNet and modifies the output size of the last linear layer in order
    # to adjust to the 58 classes of the dataset. During training all parameters are subject to backpropagation.
    if transfer == "fine_tune":
        model = models.alexnet(pretrained=True)
        model.classifier = nn.Linear(9216, 58)
    # This second model employs the same pretrained AlexNet but just the last layer is subject to backpropagation, the
    # rest of the parameters are fixed.
    elif transfer == "fixed":
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            model.classifier = nn.Linear(9216, 58)
    # This third model employs a pretrained resnet whose parameters are fixed to extract image embeddings, and then
    # employs an MLP as classifier.
    elif transfer == "resnet":
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 58)
        )

    # Sends the model to GPU if available.
    model.to(device)

    """ For assigning weights to the loss associated to each class according to the inverse of its frequency """

    # train_paths = glob.glob(os.path.join("traffic_Data", "DATA", "*", "*.png"))
    # counts = np.zeros(58)
    # for path in train_paths:
    #     class_id = int(path.split("/")[2])
    #     counts[class_id] += 1
    # weights = 1 / counts
    # weights = weights.astype(np.float32)
    # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model_savepath = os.path.join("models", "model_"+str(model_id)+"_"+str(experiment_id)+".pth")

    model, loss_list, train_acc_list, test_acc_list = train(model, criterion, optimizer, epochs, trainloader,
                                                            testloader, device, model_savepath)

    result_visualization(epochs, loss_list, train_acc_list, test_acc_list, model_id, experiment_id, save=True)

# Change for each experiment

epochs = None
lr = None
batch_size = None
model_id = None
experiment_id = None
transfer = None

main(epochs=epochs, lr=lr, batch_size=batch_size, model_id=model_id, experiment_id=experiment_id, transfer=transfer)