import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

class BilinearDecoder(nn.Module):
    def __init__(self, embedding_dim, num_relations):
        super(BilinearDecoder, self).__init__()
        self.relation_matrices = torch.nn.Parameter(torch.randn(num_relations, embedding_dim, embedding_dim))
        
    def forward(self, z, edge_index, edge_type):
        src, dst = edge_index
        rel_matrix = self.relation_matrices[edge_type]
        
        scores = (z[src] @ rel_matrix) * z[dst]
        return torch.sigmoid(scores.sum(dim=1))
    

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super(GCNModel, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, output_dim)
        self.decoder = BilinearDecoder(output_dim, num_relations)
        
    def forward(self, x, edge_index, edge_type):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_index, edge_type)
    
        
