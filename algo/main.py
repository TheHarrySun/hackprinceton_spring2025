import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from torch_geometric.transforms import RandomLinkSplit


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
node_features = torch.tensor(drug_features, dtype=torch.float32)

num_edges = len(combo2se)
edge_index = torch.zeros((2, num_edges), dtype=torch.long)
edge_type = torch.zeros((0, edge_types), dtype=torch.float32)

combos = list(combo2se.keys())
ses_from_combos = list(combo2se.values())

print(num_edges)
for i in range(num_edges):
    name1, name2 = combos[i].split('_')
    edge_index[0, i] = name2featvec[name1][0]
    edge_index[1, i] = name2featvec[name2][0]
    temp = torch.zeros(1, edge_types, dtype=torch.float32)
    for se in ses_from_combos[i]:
        temp.add(torch.tensor(se2featvec[se], dtype=torch.float32))
    edge_type = torch.cat([edge_type, temp], dim = 0)
    if (i % 1000 == 0):
        print(i)
    
print(type(edge_index))
print(type(edge_type))
print(edge_index.shape)
print(edge_type.shape)
edge_type = normalize(edge_type, axis = 1, norm = 'l1')

data = Data(x = node_features, edge_index = edge_index, edge_attr = edge_type)

model = layers.GCNModel(input_dim = len(node_features[0]), hidden_dim = len(node_features[0]) // 2, output_dim = len(node_features[0]) // 4, num_edge_types = len(edge_type[0]))

optimizer = optim.Adam(model.parameters(), lr = 0.01)

transform = RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected=False, add_negative_train_samples =False)
print(type(data))
train_data, val_data, test_data = transform(data)

def loss_function(predictions, true_edge_types):
    return F.binary_cross_entropy(predictions, true_edge_types)

def evaluate(model, data, edge_index, edge_attr):
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index)
        predicted_edge_types = (out > 0.5).long()
        loss = loss_function(out, torch.tensor(edge_attr, dtype=torch.float32))
        accuracy = (predicted_edge_types == edge_attr).float().mean().item()
    return [accuracy, loss]

def train(model, data, optimizer, epochs = 100):
    model.train()
    val_loss = []
    val_acc = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = loss_function(out, torch.tensor(data.edge_attr, dtype=torch.float32))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        val_attr = evaluate(model, val_data, val_data.edge_index, val_data.edge_attr)
        val_loss.append(val_attr[1])
        val_acc.append(val_attr[0])
    plt.plot(range(0, len(val_loss)), val_loss)
    plt.show()
    plt.plot(range(0, len(val_acc)), val_acc)
    plt.show()
    

print("amt of train_data is: ", len(train_data))
train(model, train_data, optimizer)

val_acc = evaluate(model, val_data, val_data.edge_index, val_data.edge_attr)
test_acc = evaluate(model, test_data, test_data.edge_index, test_data.edge_attr)

print(f"Val Acc: {val_acc[0]:.4f}")
print(f"Test Acc: {test_acc[0]:.4f}")

torch.save(model.state_dict(), "optimized_weights.pth")

