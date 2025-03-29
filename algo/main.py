import numpy as np
import scipy.sparse as sp

import preprocessing

val_test_size = 0.05

combo2stitch, combo2se, se2name, stitches = preprocessing.load_data()

ses = set()
for se_set in combo2se.values():
    ses = ses.union(se_set)

edge_types = len(ses)
n_drugs = len(stitches)

names = list(stitches)
drug_features = sp.identity(n_drugs).toarray()
name2featvec = dict(zip(names, drug_features))

se_features = sp.identity(edge_types).toarray()
featvec2se = dict(zip(se_features, list(ses)))

