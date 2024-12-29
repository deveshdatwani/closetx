import torch
from time import time
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, image_size=576, patch_size=16, embed_dim=768, num_heads=8, num_layers=12, num_classes=256):
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
        self.output_layer = nn.Linear(num_classes*2, 256)
        self.output_layer_2 = nn.Linear(256, 32)
        self.output_layer_3 = nn.Linear(32, 8)
        self.final_output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.patch_embedding(x) 
        x = x.flatten(2)  
        x = x.transpose(1, 2)          
        x = x + self.positional_embeddings
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x.mean(dim=1)        
        x = self.fc_out(x)

        y = self.patch_embedding(y) 
        y = y.flatten(2)  
        y = y.transpose(1, 2)          
        y = y + self.positional_embeddings
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y)
        y = y.mean(dim=1)        
        y = self.fc_out(y)
        score_dimension = torch.concat((x, y), dim=1)
        xy = self.output_layer(score_dimension)
        xy = self.output_layer_2(xy)
        xy = self.output_layer_3(xy)
        xy = self.final_output(xy)
       
        return self.sigmoid(xy)