import torch
from time import time
import torch.nn as nn
from torchvision.models import efficientnet_b0, vgg16_bn


class EfficientNet(nn.Module):
    def __init__(self, output_dim=256):
        super(EfficientNet, self).__init__()
        self.vgg16 = vgg16_bn(pretrained=True)
        self.res_block = nn.Sequential(
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512)
                                    )
        self.final_relu = nn.ReLU()
        self.fc = nn.Linear(512, output_dim)
        self.output_layer = nn.Linear(output_dim*2, 256)
        self.output_layer_2 = nn.Linear(2028, 1352)
        self.output_layer_3 = nn.Linear(1352, 768)
        self.final_output = nn.Linear(768, 128)
        self.final_output_2 = nn.Linear(128, 64)
        self.final_output_3 = nn.Linear(64, 16)
        self.pre_out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(256)
        self.forward_pass = nn.Sequential(self.output_layer_2, 
                                          self.output_layer_3, 
                                          self.final_output, 
                                          self.final_output_2, 
                                          self.final_output_3)

    def forward(self, x):
        return None
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # x = x.flatten(start_dim=1)
        # y = y.flatten(start_dim=1)
        # x = self.forward_pass(x)
        # y = self.forward_pass(y)
        # score_dimension = torch.concat((x, y), dim=1)
        # score_dimension = self.pre_out(score_dimension)
        # return self.sigmoid(score_dimension)
        # # x = self.efficient_net.features(x)
        # # residual = x
        # # x = self.res_block(x)
        # # x = self.final_relu(x + residual)
        # # x = torch.mean(x, dim=(2, 3))
        # # x = self.fc(x)
        # # y = self.efficient_net.features(y)
        # # residual = y
        # # y = self.res_block(y)
        # # y = self.final_relu(y + residual)
        # # y = torch.mean(y, dim=(2, 3))
        # # y = self.fc(y)
        # # xy = torch.sum(torch.mul(self.layer_norm(x), self.layer_norm(y)), dim=1, keepdim=True)
        # # xy = self.sigmoid(xy)
        # # print(xy)

        # # concatenation method below
        # # score_dimension = torch.concat((x, y), dim=1)
        # # xy = self.output_layer(score_dimension)
        # # xy = self.output_layer_2(xy)
        # # xy = self.output_layer_3(xy)
        # # xy = self.final_output(xy)
        # # return xy