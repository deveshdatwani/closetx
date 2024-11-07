import torch 
from torch import nn 



class FeedForwad(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, x):
        x = nn.layer1(x)
        x = nn.ReLU()(x)
        x = nn.layer2(x)