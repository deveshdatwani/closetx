import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os


def train_model(model, dataloader, criterion, optimizer, num_epochs, device, checkpoint_path=None):
    model.to(device)
    history = {'train_loss': []}
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        history['train_loss'].append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        save_checkpoint(model, optimizer, epoch, history, checkpoint_path)
    return model, history


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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (e.g., "cpu" or "cuda")')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help='Path to save the model checkpoint')
    args = parser.parse_args()
    model = "MODEL" #YourTransformerModel()  
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    dataloader = DataLoader("YourDataset()", batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    trained_model, training_history = train_model(model, dataloader, criterion, optimizer, args.epochs, args.device, args.checkpoint)


if __name__ == '__main__':
    main()