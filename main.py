##
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

# Classes associated to CNN models trained from scratch

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 58)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 1024)
        self.fc2 = nn.Linear(1024, 58)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
    loss_list = []
    train_acc_list = []
    test_acc_list = []
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
    if savepath is not None:
        torch.save(model.state_dict(), savepath)
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

def main(epochs, lr, batch_size, model_id, experiment_id):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datapath = os.path.join("traffic_Data", "DATA")
    test_datapath = os.path.join("traffic_Data", "TEST")

    tsfm_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize((0.4323, 0.4203, 0.4275),
                             (0.2423, 0.2318, 0.2463)),
        transforms.ColorJitter(),
        transforms.RandomRotation(5)

    ])
    tsfm_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize((0.4323, 0.4203, 0.4275),
                             (0.2423, 0.2318, 0.2463))
    ])

    trainset = TrainTrafficDataset(train_datapath, tsfm_train)
    testset = TestTrafficDataset(test_datapath, tsfm_test)
    print(len(trainset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = Model5()
    model.to(device)

    train_paths = glob.glob(os.path.join("traffic_Data", "DATA", "*", "*.png"))
    counts = np.zeros(58)
    for path in train_paths:
        class_id = int(path.split("/")[2])
        counts[class_id] += 1
    weights = 1 / counts
    weights = weights.astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model_savepath = os.path.join("models", "model_"+str(model_id)+"_"+str(experiment_id)+".pth")

    model, loss_list, train_acc_list, test_acc_list = train(model, criterion, optimizer, epochs, trainloader,
                                                            testloader, device, model_savepath)

    result_visualization(epochs, loss_list, train_acc_list, test_acc_list, model_id, experiment_id, save=True)

epochs = None
lr = None
batch_size = None
model_id = None
experiment_id = None

main(epochs=epochs, lr=lr, batch_size=batch_size, model_id=model_id, experiment_id=experiment_id)








