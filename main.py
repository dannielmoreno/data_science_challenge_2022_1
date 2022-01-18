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


# labels_df = pd.read_csv('labels.csv')
# class2name_dict = dict(labels_df.values)


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

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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

def evaluate(model, trainloader, testloader, device):
    correct_train, total_train, correct_test, total_test = 0, 0, 0, 0
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



def train(model, criterion, optimizer, epochs, trainloader, testloader, device, savepath=None):
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        running_loss = 0.0
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


def main(epochs, batch_size, model_id, experiment_id):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tsfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize((0.4323, 0.4203, 0.4275),
                             (0.2423, 0.2318, 0.2463))
    ])

    train_datapath = os.path.join("traffic_Data", "DATA")
    test_datapath = os.path.join("traffic_Data", "TEST")

    trainset = TrainTrafficDataset(train_datapath, tsfm)
    testset = TestTrafficDataset(test_datapath, tsfm)
    print(len(trainset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = Model1()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_savepath = os.path.join("models", "model_"+str(model_id)+"_"+str(experiment_id)+".pth")

    model, loss_list, train_acc_list, test_acc_list = train(model, criterion, optimizer, epochs, trainloader,
                                                            testloader, device, model_savepath)

    result_visualization(epochs, loss_list, train_acc_list, test_acc_list, model_id, experiment_id, save=True)

main(epochs=10, batch_size=64, model_id=1, experiment_id=5)








