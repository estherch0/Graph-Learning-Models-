# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import Data
import torch_geometric
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

# Set random seed for reproducibility
torch_geometric.seed_everything(50)

# GCN for Node Classification
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes): 
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # log softmax cause it's useful for multiclass classification 
        return F.log_softmax(x, dim=1)

# changed to max pool 
# looks like the model for graph classification does well for multi-class data (like the ENZYME) but not so much for the binary one 
# let's try making the model deeper and increase size of hidden layers - didn't work (later we show increasing layers don't help)
# let's try making the model use global max instead - highered accuracy for ENZYME dataset 
# let's try cross entrophy loss - better results (especially for ENZYME)
# looking at the loss changes in IMDB, there is barely any changes, so let's trying lowering learning rate - increased accuracy by a LOT


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, 16)  # first GCN layer
        self.conv2 = GCNConv(16, 16)            # second GCN layer
        # fully connected layer for classification
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # recognized the these nodes don't have features 
        if x is None:
            x = torch.ones((edge_index.max().item() + 1, 1)) 
            
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)  # aggregate node features to graph level
        return F.log_softmax(self.fc(x), dim=1)  # log-softmax for multi-class classification

class GCNGraphClassifierIMDB(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# since the training accuracy looks a little high, let's normalize it somehow
# what about adding batchnorm? - didn't work
# what about learning rate? - didn't change much - let's try dropout
class GINModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(num_features, 64))
        self.conv2 = GINConv(torch.nn.Linear(64, num_classes))
        self.dropout = torch.nn.Dropout(0.5) # added
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Simple GINGraphClassifier
class GINGraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINGraphClassifier, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(num_features, 16))
        self.conv2 = GINConv(torch.nn.Linear(16, 16))
        self.fc = torch.nn.Linear(16, num_classes)
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones((edge_index.max().item() + 1, 1)) 

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.pool(x, batch)
        x = self.fc(x)
        return x
    

# Simple two layed GAT for node classification 
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=num_heads)
        self.conv2 = GATConv(16 * num_heads, num_classes, heads=1, concat=False)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# see multi-layers
class GAT_MultiLayer(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=16, num_heads=1):
        super(GAT_MultiLayer, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(num_features, hidden_dim, heads=num_heads))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        self.layers.append(GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# SO let's do 4 layers:
# did some dropout experimenting to regularize the model
class EnhancedGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_heads=6):
        super(EnhancedGAT, self).__init__() 
        self.conv1 = GATConv(num_features, 16, heads=num_heads, concat=True)
        self.conv2 = GATConv(16 * num_heads, 16, heads=num_heads, concat=True)
        self.conv3 = GATConv(16 * num_heads, 16, heads=num_heads, concat=True)
        self.conv4 = GATConv(16 * num_heads, num_classes, heads=1, concat=False)  
        self.dropout = torch.nn.Dropout(0.4)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # pass through each GAT layer with activation and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        
        return F.log_softmax(x, dim=1) 

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_heads=8, hidden_dim=32):
        super(GATGraphClassifier, self).__init__()
        # two layer gat
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.num_features = num_features
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.randn((edge_index.max().item() + 1, self.num_features))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.fc(x)
        return x

# Long Range Benchmark Models (PeptideStruct)
class GCNPeptideStruct(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNPeptideStruct, self).__init__()
        
        # Define the GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.6) # regularize
        self.fc = torch.nn.Linear(hidden_channels, 11)  
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        edge_index = edge_index.to(torch.int64)  # convert edge_index to int64 (Long)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
        else:
            edge_weight = edge_weight.to(torch.float32)  # ensure edge_weight is float
        
        x = x.to(torch.float32)  # convert x to float32 if it's not already
        
        # GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # third GCN layer
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # graph-level pooling to get graph-level features
        
        # fully connected layer
        x = self.fc(x)
        
        return x

class GINPeptideStruct(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINPeptideStruct, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(num_features, 64))
        self.conv2 = GINConv(torch.nn.Linear(64, 128))
        self.conv3 = GINConv(torch.nn.Linear(128, 64))
        self.fc = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5) # regularize

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GATPeptideStruct(torch.nn.Module):
    def __init__(self, num_features, num_classes, heads=4):
        super(GATPeptideStruct, self).__init__()
        self.conv1 = GATConv(in_channels=num_features, out_channels=64, heads=heads)
        self.conv2 = GATConv(in_channels=64 * heads, out_channels=128, heads=heads)
        self.conv3 = GATConv(in_channels=128 * heads, out_channels=64, heads=1)  # Use 1 head for final layer
        self.fc = torch.nn.Linear(64, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5) # regularize

    def forward(self, x, edge_index, batch, edge_weight=None):
        # ensure edge_index is of type int64
        edge_index = edge_index.to(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
        else:
            edge_weight = edge_weight.to(torch.float32)  # Ensure edge_weight is float
        
        x = x.to(torch.float32) 
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

models = {
    'GCN': GCN,
    'GCNGraphClassifier': GCNGraphClassifier,
    'GCNGraphClassifierIMDB': GCNGraphClassifierIMDB,
    'GINModel': GINModel,
    'GINGraphClassifier': GINGraphClassifier,
    'GAT': GAT,
    'EnhancedGAT': EnhancedGAT,
    'GATGraphClassifier': GATGraphClassifier,
    'GAT_MultiLayer': GAT_MultiLayer,
    "GCNPeptideStruct": GCNPeptideStruct,
    "GINPeptideStruct": GINPeptideStruct,
    "GATPeptideStruct": GATPeptideStruct
}
