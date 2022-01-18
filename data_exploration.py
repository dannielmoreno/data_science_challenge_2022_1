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

tsfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize((0.4323, 0.4203, 0.4275),
                             (0.2423, 0.2318, 0.2463))
])

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
        class_id = int(img_path.split('/')[2])
        return image, class_id

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

train_datapath = os.path.join("traffic_Data", "DATA")
trainset = TrainTrafficDataset(train_datapath, tsfm)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
mean, std = get_mean_and_std(trainloader)
print(mean)
print(std)
