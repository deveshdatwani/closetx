import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder.resnets import vgg_16
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


def get_image(url):
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])
                                    ])
    image = transform(Image.open(url)).unsqueeze(0)
    return image


def get_input(i):
    if i % 2 == 0:
        url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top"
        idx = 2
        image_path = os.path.join(url, os.listdir(url)[idx]) 
        top = get_image(image_path)
        url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/bottom"
        idx = 2
        image_path = os.path.join(url, os.listdir(url)[idx])
        bottom = get_image(image_path)
        y = torch.tensor([[1]], dtype=torch.float32)
        return top, bottom, y
    else:
        url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top"
        idx = 2
        image_path = os.path.join(url, os.listdir(url)[idx]) 
        top = get_image(image_path)
        url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/bottom"
        idx = 3
        image_path = os.path.join(url, os.listdir(url)[idx])
        bottom = get_image(image_path)
        y = torch.tensor([[0]], dtype=torch.float32)
        return top, bottom, y
     

def train():
    model = vgg_16()
    criterion = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for i in range(20):
        x1, x2, y = get_input(i)
        opt.zero_grad()
        out = model(x1, x2)
        loss = criterion(out, y)
        print(f"loss: {loss}")
        loss.backward()
        opt.step()


if __name__ == "__main__":
    train()