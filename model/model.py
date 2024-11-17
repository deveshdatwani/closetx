import torch
import torch.nn as nn
import torch.nn.functional as F


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