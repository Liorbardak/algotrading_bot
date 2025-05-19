import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# === Define the model ===
class VariableInputCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        # x = self.conv(x)
        x = self.conv1(x)  # batch ,depth, width, height
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# === Instantiate the model ===
model = VariableInputCNN(num_classes=10)


# === Generate a batch of noisy images with random sizes ===
def generate_noisy_input(batch_size=1, min_size=32, max_size=128):
    H = random.randint(min_size, max_size)
    W = random.randint(min_size, max_size)
    image = torch.randn(batch_size, 3, H, W)  # Noise image
    return image


# === Run inference ===
model.eval()
with torch.no_grad():
    input_tensor = generate_noisy_input()
    print(f"Input shape: {input_tensor.shape}")  # Variable size input
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
