import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F

import preprocessing
import layers

val_test_size = 0.05

combo2stitch, combo2se, se2name, stitches = preprocessing.load_data()

ses = set()
for se_set in combo2se.values():
    ses = ses.union(se_set)

edge_types = len(ses)
n_drugs = len(stitches)

names = list(stitches)
drug_features = sp.identity(n_drugs).toarray()

counter = 0
for i in range(len(names)):
    names[i] = (counter, names[i])
    counter += 1

name2featvec = dict(zip(names, drug_features))

se_features = sp.identity(edge_types).toarray()
num2se = dict(zip(range(0, edge_types), list(ses)))

node_features = torch.tensor(drug_features)
print(node_features.shape)


# model = layers.GCNModel()