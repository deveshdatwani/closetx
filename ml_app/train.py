import os
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
from dataloader import *
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder.type_specific_model import FullModel
warnings.filterwarnings("ignore")


def train_model(tsn_model, tp_model, dataloader, optim, criterion, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_data = {
            "epoch" : 0,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "loss": 0
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint created at {checkpoint_path}")
    else:
        checkpoint = torch.load(checkpoint_path)
    for top, bottom, wrong_bottom  in dataloader:
        top = tsn_model(top)
        bottom = tsn_model(bottom)
        wrong_bottom = tsn_model(bottom)
        top_embedding = tp_model(top)
        bottom_embedding = tp_model(bottom)
        wrong_bottom_embedding = tp_model(wrong_bottom)
        tsn_model.zero_grad()
        tp_model.zero_grad()
        loss = criterion(top_embedding, bottom_embedding, wrong_bottom_embedding)
        loss.backward()
        optim.step()
    return None


def save_checkpoint(model, optimizer, epoch, history, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="training hyperparameters")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (e.g., "cpu" or "cuda")')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help='Path to save the model checkpoint')
    args = parser.parse_args()
    custom_dataset = CustomDataset(data_path="/home/deveshdatwani/closetx/ml_app/models/dataset",
                                raw_path="/home/deveshdatwani/closetx/ml_app/models/dataset", 
                                top_path="/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top", 
                                bottom_path="/home/deveshdatwani/closetx/ml_app/models/dataset/positive/bottom")
    dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.BCELoss()
    

def _main():
    dataloader = DataLoader(CustomDataset())
    model = FullModel()
    criterion = nn.TripletMarginLoss()
    opt = optim.AdamW(params=model.parameters(), lr=0.0001)
    for i in range(10):
        for top_img, bottom_img, wrong_bottom_img in dataloader:
            top_output = model(top_img)
            bottom_output = model(bottom_img)
            wrong_output = model(wrong_bottom_img)
            loss = criterion(top_output, bottom_output, wrong_output)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"loss: {loss}") 


if __name__ == '__main__':
    _main()