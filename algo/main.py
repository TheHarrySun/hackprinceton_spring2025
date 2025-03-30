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

num_edges = len(combo2se)
edge_index = torch.zeros((2, num_edges))
edge_type = torch.zeros((0, edge_types))

combos = list(combo2se.keys())
ses_from_combos = list(combo2se.values())

for i in range(num_edges):
    name1, name2 = combos[i].split('_')
    edge_index[0, i] = name2featvec[name1][0]
    edge_index[1, i] = name2featvec[name2][0]
    temp = torch.zeros(1, edge_types)
    for se in ses_from_combos[i]:
        temp.add(torch.tensor(se2featvec[se]))
    edge_type = torch.cat([edge_type, temp], dim = 0)
    
print(type(edge_index))
print(type(edge_type))
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
