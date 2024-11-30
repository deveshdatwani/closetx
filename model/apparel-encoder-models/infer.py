import torch
from model import EfficientNet

if __name__ == '__main__':
    _model = EfficientNet()
    input_img = torch.rand(1, 3, 576, 576)
    output = _model(input_img)