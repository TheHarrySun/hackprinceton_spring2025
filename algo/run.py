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
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split

import preprocessing
import layers
import ffnn

combo2stitch, combo2se, se2name, stitches = preprocessing.load_data("../bio-decagon-combo-mini.csv")

cids = [int(x[3:]) for x in stitches]

smiles_dict = preprocessing.get_smiles_from_cids(cids)

vec_dict = preprocessing.extract_molecular_features(smiles_dict)

vec_len = len(list(vec_dict.values())[0])
print("length of molecular descriptor vector: ", vec_len)

drug1 = []
drug2 = []
possible_ses = {}
counter = 0
for key1, val1 in vec_dict.items():
    for key2, val2 in vec_dict.items():
        if (key1 == key2):
            continue
        name = "CID" + str(key1) + "_" + "CID" + str(key2)
        if (name in combo2se):
            drug1.append(key1)
            drug2.append(key2)
            for se in combo2se[name]:
                if se not in possible_ses:
                    possible_ses[se] = counter
                    counter += 1
                
print(drug1)
print(drug2)
labels = []
for i in range(len(drug1)):
    name = "CID" + str(drug1[i]) + "_" + "CID" + str(drug2[i])
    total_ses = len(possible_ses)
    label = torch.zeros((1, total_ses))
    for se in combo2se[name]:
        label.add(F.one_hot(possible_ses[se], total_ses))
    labels.append(label)
    print(labels)
labels = normalize(labels, axis = 1, norm = 'l1')
    
drug1_train, drug1_test, drug2_train, drug2_test, label_train, label_test = train_test_split(drug1, drug2, label, test_size = 0.1, random_state = 42)

model = ffnn.FFNN(len(drug1_train[0]), hidden_dim = (len(drug1_train[0])) // 2, combined_hidden = (len(drug1_train[0])) // 4, num_classes = counter + 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    outputs = model(drug1_train, drug2_train)
    loss = criterion(outputs, label_train)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
model.eval()
with torch.no_grad():
    outputs = model(drug1_test, drug2_test)
    loss = criterion(outputs, label_test)
    
print(f"Test MSE Loss: {loss.item():.4f}")