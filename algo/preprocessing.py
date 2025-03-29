from collections import defaultdict
import pandas as pd


def load_data(fname = "data/bio-decagon-combo.csv"):
    data = pd.read_csv(fname)
    cid1 = data["STITCH 1"]
    cid2 = data["STITCH 2"]
