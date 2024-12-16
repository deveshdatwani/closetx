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
        self.output_layer = nn.Linear(output_dim*2, 1)

    def forward(self, x, y):
        x = self.efficient_net.features(x)
        residual = x
        x = self.res_block(x)
        x = self.final_relu(x + residual)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        y = self.efficient_net.features(y)
        residual = y
        y = self.res_block(y)
        y = self.final_relu(y + residual)
        y = torch.mean(y, dim=(2, 3))
        y = self.fc(y)
        score_dimension = torch.concat((x, y), dim=1)
        score = self.output_layer(score_dimension)
        return score
    

class VisionTransformer(nn.Module):
    def __init__(self, image_size=576, patch_size=16, embed_dim=768, num_heads=8, num_layers=12, num_classes=756):
        super(VisionTransformer, self).__init__()        
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim        
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)        
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4, 
                dropout=0.1
            ) for _ in range(num_layers)
        ])        
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x) 
        x = x.flatten(2)  
        x = x.transpose(1, 2)          
        x = x + self.positional_embeddings
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x.mean(dim=1)        
        x = self.fc_out(x)        
        return x