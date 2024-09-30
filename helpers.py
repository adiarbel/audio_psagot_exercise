import numpy as np
import torch
# Converts list of the top-k class predictions to top-k corresponding labels
def convert_to_labels(preds, index_to_class, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([index_to_class[i] for i in idx[:k]]))

    return ans, ids


def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

