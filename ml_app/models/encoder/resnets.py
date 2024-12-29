import torch
from time import time
import torch.nn as nn
from torchvision.models import efficientnet_b0, vgg16_bn, resnet34, resnet152


class vgg_16_cust(nn.Module):
    def __init__(self, output_dim=256):
        super(vgg_16_cust, self).__init__()
        self.vgg_16 = resnet152(pretrained=True)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Sequential(nn.Linear(2000, 1),)
    
    def forward(self, top, bottom):
        top = self.vgg_16(top)
        bottom = self.vgg_16(bottom)
        out = torch.concatenate((top, bottom), dim=1)
        output = self.output_layer(out)
        return self.sigmoid(output)