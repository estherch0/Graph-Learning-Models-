import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import models # Import the dictionary of models
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data
import torch_geometric
from sklearn.model_selection import train_test_split
from train_utils import (
    train_and_evaluate_node_classification,
    train_and_evaluate_IMDB_GCN,
    train_and_evaluate,
    train_and_evaluate_gat,
    split_dataset,
    train_and_evaluate_peptides_struct,
    plot_losses
)

torch_geometric.seed_everything(50)

# Load model parameters from data-params.json
with open('data-params.json') as f:
    data_params = json.load(f)


# Define datasets
datasets = {
    'Cora': lambda: Planetoid(root='data/Cora', name='Cora'),
    'ENZYMES': lambda: TUDataset(root='data/ENZYMES', name='ENZYMES'),
    'IMDB-BINARY': lambda: TUDataset(root='data/IMDB-BINARY', name='IMDB-BINARY'),
    'IMDB': lambda: TUDataset(root='data/IMDB-BINARY', name='IMDB-BINARY'),
    'Peptides-struct': lambda: LRGBDataset(root='data/peptides-struct', name='peptides-struct')
}

def run_model(model_name, model_class, config, config_index=None):
    # Load dataset
    torch_geometric.seed_everything(50)
    dataset_name = config['dataset']
    dataset = datasets[dataset_name]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_type = 'node' if dataset_name == 'Cora' else 'graph'


    # Special handling for GAT_MultiLayer with varying layer counts
    if model_name == "GAT_MultiLayer":
        layer_counts = config.get("layer_counts", [2, 3, 4, 5])
        results = train_and_evaluate_gat(dataset, device, layer_counts=layer_counts)
        
        # Save the accuracy for each layer count
        results_dir = f"results/{model_name}_{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/layer_accuracies.txt", "w") as f:
            for layers, accuracy in results:
                f.write(f"Layers: {layers}, Test Accuracy: {accuracy:.4f}\n")
        
        # Plot and save the accuracy vs. layer count graph
        layer_counts, accuracies = zip(*results)
        plt.plot(layer_counts, accuracies, marker='o')
        plt.xlabel("Number of Layers")
        plt.ylabel("Test Accuracy")
        plt.title(f"{model_name} Test Accuracy vs. Number of Layers")
        plt.savefig(f"{results_dir}/layer_accuracy_plot.png")
        # plt.show()
        print(f"Results for {model_name} on {dataset_name} saved in {results_dir}")
        return

    # Split dataset if graph classification
    if data_type == 'graph':
        train_data, test_data = split_dataset(dataset)
    else:
        data = dataset[0].to(device)

    # Initialize model
    if (dataset_name == 'IMDB' or dataset_name == 'IMDB-BINARY'):
        model = model_class(num_features=1, num_classes=2).to(device)
    elif dataset_name == 'Peptides-struct':
        if model_name == "GCNPeptideStruct":
            model = model_class(config['in_channels'], config['hidden_channels'], config['out_channels']).to(device)
        else:
            model = model_class(dataset.num_node_features, dataset.num_classes).to(device)
    else:
        model = model_class(dataset.num_node_features, dataset.num_classes).to(device)

    # Training and evaluation
    if data_type == 'node':
        train_acc, test_acc, train_losses = train_and_evaluate_node_classification(
            model, data, device, config['loss_fn'], config['epochs'], config['lr'], config['weight_decay']
        )
    elif dataset_name == 'Peptides-struct':
        train_losses, test_mse, avg_epoch_loss = train_and_evaluate_peptides_struct(model=model, train_dataset=train_data, test_dataset = test_data, device=device, epochs=config['epochs'], lr = config['lr'], batch_size = config["batch_size"])
    elif (dataset_name == 'IMDB' or dataset_name == 'IMDB-BINARY') & (model_name == "GCNGraphClassifierIMDB"):
        train_acc, test_acc, train_losses = train_and_evaluate_IMDB_GCN(dataset, model = model)
    else:
        train_acc, test_acc, train_losses = train_and_evaluate(
            model, train_data, test_data, device, config['loss_fn'], config['epochs'], config['lr'], config['weight_decay']
        )

    # Save results to directory
    config_suffix = f"_{config_index}" if config_index is not None else ""
    results_dir = f"results/{model_name}_{dataset_name}{config_suffix}"
    os.makedirs(results_dir, exist_ok=True)

    if dataset_name == 'Peptides-struct':
        with open(f"{results_dir}/accuracies.txt", "w") as f:
            f.write(f"Final Training MSE: {avg_epoch_loss:.4f}\n")
            f.write(f"Final Testing MSE: {test_mse:.4f}\n")
    else:
    # Save accuracy results
        with open(f"{results_dir}/accuracies.txt", "w") as f:
            f.write(f"Final Training Accuracy: {train_acc:.4f}\n")
            f.write(f"Final Testing Accuracy: {test_acc:.4f}\n")

    # Save training loss plot
    plot_losses(train_losses, save_path=f"{results_dir}/training_loss.png")
    print(f"Results for {model_name} on {dataset_name} saved in {results_dir}")

if __name__ == "__main__":
    for model_name, model_class in models.items():
        if model_name in data_params:
            configs = data_params[model_name]
            # Ensure configs is always a list for iteration
            if not isinstance(configs, list):
                configs = [configs]  # Wrap single dictionary in a list

            # Run each configuration for the model
            for idx, config in enumerate(configs):
                # Check if 'loss_fn' exists in config before accessing it
                if 'loss_fn' in config:
                    # Map loss function names to actual functions
                    if config['loss_fn'] == 'nll_loss':
                        config['loss_fn'] = F.nll_loss
                    elif config['loss_fn'] == 'cross_entropy':
                        config['loss_fn'] = torch.nn.CrossEntropyLoss()
                    elif config['loss_fn'] == 'mse_loss':
                        config['loss_fn'] = torch.nn.MSELoss()
                
                print(f"Running {model_name} (Config {idx})...")
                run_model(model_name, model_class, config, config_index=idx)

