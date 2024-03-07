import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

def trainer(model, train_loader, optimizer, criterion, num_epochs, device):
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return losses

def export_Plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input_size', type=int, default=64, help='Input size of the model')
    parser.add_argument('--output_size', type=int, default=1, help='Output size of the model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    input_size = args.input_size
    output_size = args.output_size
    num_epochs = args.num_epochs
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model = ...

# TODO:
# - model architecture
# - confusion_matrix
# - parallel training
# - etc