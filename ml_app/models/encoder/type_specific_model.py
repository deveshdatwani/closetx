import torch
from time import time
import torch.nn as nn
from torchvision.models import resnet34

class TypeProjectModel(nn.Module):
    def __init__(self, input_dim=1000, output_dim=512):
        super(TypeProjectModel, self).__init__()
        self.fc = nn.Sequential(
                                nn.Linear(input_dim, 512),
                                nn.Linear(512, 512),
                                nn.Linear(512, output_dim)
                                )
    
    def forward(self, x):
        return self.fc(x)


class TypeSpecificModel(nn.Module):
    def __init__(self, inner_dim=256, type_specific_network=None):
        super(TypeSpecificModel, self).__init__()
        self.backbone = backbone_network=resnet34(pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.model_1 = TypeProjectModel()
        self.model_2 = TypeSpecificModel()

    def forward(self, x):
        x = self.model_2(x)
        x = self.model_1(x)
        return x