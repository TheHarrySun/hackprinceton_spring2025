import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import os

# Data preprocessing
def load_snap_data(edge_list_path, features_path=None):
    edges = pd.read_csv(edge_list_path, sep=',', header=None)
    edges.columns = ['drug1', 'drug2', 'se_id', 'se_name']

    G = nx.from_pandas_edgelist(edges, 'drug1', 'drug2')
    adj_matrix = nx.adjacency_matrix(G)

    if features_path:
        features = pd.read_csv(features_path, index_col=0).values
    else:
        features = np.eye(adj_matrix.shape[0])  # Identity matrix if no features

    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data

# GCN Model class
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x)

# Training and evaluation
def train(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index).squeeze()
        labels = torch.ones(data.x.shape[0])  # Assuming edges represent interactions

        loss = F.binary_cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model/best_gcn_model.pth')

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).squeeze().cpu().numpy()
        labels = np.ones(data.x.shape[0])  # Labels for existing edges (positive class)

        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        print(f'AUC: {auc:.4f}, AP: {ap:.4f}')

# Main function
def main():
    os.makedirs('model', exist_ok=True)

    data = load_snap_data('../bio-decagon-combo-mini.csv')  # Provide correct path to your dataset

    model = GCN(num_features=data.x.shape[1])
    train(model, data)
    evaluate(model, data)

if __name__ == '__main__':
    main()
