import torch
import torchvision.models as models
from torchvision.io import read_image
from torchvision import transforms
import os
import torch.nn as nn

class Model():
    def __init__(self):
        self.model = models.alexnet()
        self.model.classifier = nn.Linear(9216, 58)
        self.model.load_state_dict(torch.load(os.path.join("models", "best_model.pth")))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    def predict(self, img_path):
        image = read_image(img_path) / 255
        image = self.transforms(image)
        image = image.unsqueeze(0)
        output = self.model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()




