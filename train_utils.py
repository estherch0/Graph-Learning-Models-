import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import models  # Import the dictionary of models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch_geometric
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from models import GAT_MultiLayer 

# Set random seed for reproducibility
torch_geometric.seed_everything(50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Loading Functions
def load_planetoid_dataset(name='Cora', root='/tmp/Cora'):
    return Planetoid(root=root, name=name)

def load_tudataset(name='ENZYMES', root='data'):
    return TUDataset(root=root, name=name)
    
def train_and_evaluate_gat(dataset, device, layer_counts, hidden_dim=16, num_heads=1, lr=0.001, weight_decay=5e-4, epochs=100):
    torch_geometric.seed_everything(50)
    data = dataset[0].to(device)
    results = []

    for layers in layer_counts:
        model = GAT_MultiLayer(num_features=data.x.size(1), num_classes=dataset.num_classes, num_layers=layers, hidden_dim=hidden_dim, num_heads=num_heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        pred = model(data).argmax(dim=1)
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(train_correct) / int(data.train_mask.sum())
        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(data.test_mask.sum())
        
        print(f'Layers: {layers}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        results.append((layers, test_acc))

    # Plotting the results
    layer_counts, accuracies = zip(*results)
    plt.plot(layer_counts, accuracies, marker='o')
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.title("GAT Accuracy vs. Number of Layers")
    # plt.show()

    return results


# Dataset Loading Functions
def load_planetoid_dataset(name='Cora', root='/tmp/Cora'):
    return Planetoid(root=root, name=name)

def load_tudataset(name='ENZYMES', root='data'):
    return TUDataset(root=root, name=name)

# Split dataset for graph classification
def split_dataset(dataset, test_size=0.2, seed=42):
    dataset = dataset.shuffle()
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]
    return train_data, test_data

# Training function for node classification
def train_and_evaluate_node_classification(model, data, device, criterion=F.nll_loss, epochs=50, lr=0.01, weight_decay=5e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    model.eval()
    pred = model(data).argmax(dim=1)
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = int(train_correct) / int(data.train_mask.sum())
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())
    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

    return train_acc, test_acc, train_losses


def train_and_evaluate(model, train_data, test_data, device, criterion = F.nll_loss, epochs=50, lr=0.01, weight_decay=5e-4):
    torch_geometric.seed_everything(50)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        train_losses.append(total_loss)
    model.eval()
    train_correct = 0
    train_total = 0
    for data in train_data:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        train_correct += (pred == data.y).sum().item()
        train_total += data.y.size(0)
    train_acc = train_correct / train_total

    # calculate test accuracy
    test_correct = 0
    test_total = 0
    for data in test_data:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        test_correct += (pred == data.y).sum().item()
        test_total += data.y.size(0)
    test_acc = test_correct / test_total

    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    return train_acc, test_acc, train_losses

def train_and_evaluate_peptides_struct(model, train_dataset, test_dataset, device, batch_size, epochs, lr):
    scaler = StandardScaler()

    # function to scale each graph's node features
    def scale_features(dataset):
        for data in dataset:
            # reshape data.x if it's not already 2D
            if data.x.ndimension() == 1:
                data.x = data.x.unsqueeze(1)  # add dummy feature dimension if necessary
            data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)

    # scale features in train and test datasets
    scale_features(train_dataset)
    scale_features(test_dataset)
    # dataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # forward pass
            out = model(x, edge_index, batch)  # pass these three inputs to the model
            
            target = data.y  # ground truth
            
            # compute loss 
            loss = criterion(out, target)
            
            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Track average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")
    
    model.eval()
    test_mse = 0
    
    for data in test_loader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        with torch.no_grad():
            out = model(x, edge_index, batch)
        
        target = data.y
        test_mse += torch.nn.MSELoss()(out, target).item()  # mean squared error for regression
    
    test_mse /= len(test_loader)
    print(f'Train MSE: {avg_epoch_loss:.4f}, Test MSE: {test_mse:.4f}')
    return train_losses, test_mse, avg_epoch_loss

# Plotting function for training losses
def plot_losses(train_losses, save_path=None):
    torch_geometric.seed_everything(50)
    """Plot training loss over epochs and optionally save the plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Training loss plot saved to {save_path}")