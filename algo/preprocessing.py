from collections import defaultdict
import requests
from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np

def load_data(fname = "../bio-decagon-combo.csv"):
    data = open(fname)
    combo_to_stitch = {}
    combo_to_se = defaultdict(set)
    se_to_name = {}
    data.readline()
    stitches = set()
    for line in data:
        stitch1, stitch2, se, se_name = line.strip().split(',')
        combo = stitch1 + '_' + stitch2
        stitches.add(stitch1)
        stitches.add(stitch2)
        combo_to_stitch[combo] = [stitch1, stitch2]
        combo_to_se[combo].add(se)
        se_to_name[se] = se_name
    data.close()
    return combo_to_stitch, combo_to_se, se_to_name, stitches

def get_smiles_from_cids(cids):
    smiles_dict = {}
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/CanonicalSMILES/JSON"

    for cid in cids:
        response = requests.get(base_url.format(cid))
        
        if response.status_code == 200:
            data = response.json()
            try:
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                smiles_dict[cid] = smiles
            except (KeyError, IndexError):
                smiles_dict[cid] = None
        else:
            smiles_dict[cid] = None
    
    return smiles_dict

def extract_molecular_features(smiles_dict):
    vec_dict = {}
    for key, val in smiles_dict.items():
        if (val == None):
            vec_dict[key] = None
            continue
        mol = Chem.MolFromSmiles(val)
        if mol is None:
            raise ValueError("Invalid SMILES")
    
        calc = Calculator(descriptors, ignore_3D=True)
        descriptor_values = calc(mol)
        descriptor_values = [float(d) if d is not None else np.nan for d in descriptor_values]
        descriptor_values = np.nan_to_num(descriptor_values, nan=0.0)
        feature_vector = np.array(descriptor_values)
        vec_dict[key] = feature_vector
        if (len(feature_vector) != 1613):
            print("length doesn't match")
    return vec_dict