from PIL import Image
import numpy as np
from time import time
import cv2
import torch 
from collections import OrderedDict
from models.apparel_encoder_models.model import EfficientNet

def inference(model, top, bottom):
    top = Image.open(top)
    top = np.array(top, np.float32)
    top = cv2.resize(top,(576, 576))
    top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
    bottom = Image.open(bottom)
    bottom = np.array(bottom, np.float32)
    bottom = cv2.resize(bottom,(576, 576))
    bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
    top = torch.permute(torch.from_numpy(top), (2, 0, 1))
    bottom = torch.permute(torch.from_numpy(bottom), (2, 0, 1))
    top = top.unsqueeze(0)
    bottom = bottom.unsqueeze(0)
    start_time = time()
    score = model(bottom, top)
    end_time = time()
    #current_app.logger.info(f"Inferenced {top.shape[0]} batches in {end_time - start_time} seconds")
    return score


def get_model():
    model = EfficientNet()
    new_state_dict = OrderedDict()
    model.load_state_dict(torch.load("/home/deveshdatwani/check.pth")["model_state_dict"])
    return model

model = get_model()
top = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top/cn55717014.jpg" 
bottom = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/bottom/cn56200152.jpg"

print(inference(model, top, bottom))
