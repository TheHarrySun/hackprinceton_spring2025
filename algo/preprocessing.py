from collections import defaultdict

def load_data(fname = "../bio-decagon-combo.csv"):
    data = open(fname)
    combo_to_stitch = {}
    combo_to_se = defaultdict(set)
    se_to_name = {}
    data.readline()
    for line in data:
        stitch1, stitch2, se, se_name = line.strip().split(',')
        combo = stitch1 + '_' + stitch2
        combo_to_stitch[combo] = [stitch1, stitch2]
        combo_to_se[combo].add(se)
        se_to_name[se] = se_name
    data.close()
    return combo_to_stitch, combo_to_se, se_to_name
