import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import normalize
from torch_geometric.utils import train_test_split_edges, negative_sampling

import preprocessing
import layers

combo2stitch, combo2se, se2name, stitches = preprocessing.load_data()

ses = set()
for se_set in combo2se.values():
    ses = ses.union(se_set)

edge_types = len(ses)
n_drugs = len(stitches)

names = list(stitches)
drug_features = sp.identity(n_drugs).toarray()

counter = 0
drug_feat_count = []
for i in range(len(drug_features)):
    drug_feat_count.append((counter, drug_features[i]))
    counter += 1

name2featvec = dict(zip(names, drug_feat_count))

se_features = sp.identity(edge_types).toarray()
num2se = dict(zip(range(0, edge_types), list(ses)))

se2featvec = dict(zip(list(ses), se_features))
node_features = torch.tensor(drug_features)
print(node_features.shape)

num_edges = len(combo2se)
edge_index = torch.empty((2,0), dtype=torch.
edge_type = torch.empty(
for key, val in combo2se.items():
    name1, name2 = key.split('_')
    edge_index[0].append(name2featvec[name1][0])
    edge_index[1].append(name2featvec[name2][0])
    temp = torch.zeros(se2featvec[list(val)[0]].shape)
    for se in val:
        temp.add(torch.tensor(se2featvec[se]))
    edge_type.append(temp)
    
print(type(edge_index))
print(type(edge_type))
edge_index = np.array(edge_index)
edge_type = np.array(edge_type)
print(edge_index.shape)
print(edge_type.shape)
print(edge_type)
edge_type = normalize(edge_type, axis = 1, norm = 'l1')

data = Data(x = node_features, edge_index = edge_index, edge_attr = edge_type)

model = layers.GCNModel(input_dim = len(node_features[0]), hidden_dim = len(node_features[0]) // 2, output_dim = len(node_features[0]) // 4, num_edge_types = len(edge_type[0]))

optimizer = optim.Adam(model.parameters(), lr = 0.01)

data = train_test_split_edges(data, test_ratio=0.1, val_ratio = 0.1)

train_edges = data.train_edge_index
val_edges = data.val_edge_index
test_edges = data.test_edge_index

def loss_function(predictions, true_edge_types):
    return F.binary_cross_entropy(predictions, true_edge_types.float())

def train(model, data, optimizer, epochs = 100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        out = model(data.x, data.train_edge_index, data.train_edge_attr)
        
        loss = loss_function(out, data.train_edge_attr)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            
train(model, data, optimizer)

def evaluate(model, data, edge_index, edge_attr):
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index, edge_attr)
        predicted_edge_types = (out > 0.5).long()
        accuracy = (predicted_edge_types == edge_attr).float().mean().item()
    return accuracy

val_acc = evaluate(model, data, data.val_edge_index, data.val_edge_attr)
test_acc = evaluate(model, data, data.test_edge_index, data.test_edge_attr)

print(f"Val Acc: {val_acc:.4f}")
print(f"Test Acc: {test_acc:.4f}")
