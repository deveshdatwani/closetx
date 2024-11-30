import torch
from time import time
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNet(nn.Module):
    def __init__(self, output_dim=256):
        super(EfficientNet, self).__init__()
        self.efficient_net = efficientnet_b0(pretrained=True)
        self.efficient_net.features[0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.res_block = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
            nn.BatchNorm2d(1280)
        )
        self.final_relu = nn.ReLU()
        self.fc = nn.Linear(1280, output_dim)


    def forward(self, x):
        x = self.efficient_net.features(x)
        residual = x
        x = self.res_block(x)
        x = self.final_relu(x + residual)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x 