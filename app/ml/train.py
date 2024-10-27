import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    history = {'train_loss': []}
    for epoch in range(num_epochs):
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
    return model, history


def main():
    parser = argparse.ArgumentParser(description="training hyperparameters")    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (e.g., "cpu" or "cuda")')
    args = parser.parse_args()
    # model = YourTransformerModel()  # Replace with your model
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # dataloader = DataLoader(YourDataset(), batch_size=args.batch_size, shuffle=True)  # Replace with your dataset
    # trained_model, training_history = train_model(model, dataloader, criterion, optimizer, args.epochs, args.device)


if __name__ == '__main__':
    main()