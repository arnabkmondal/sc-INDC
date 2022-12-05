import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import cuda
import h5py
import scanpy as sc
import scipy as sp
import pandas as pd

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mouse_ES_cell')
parser.add_argument('--nb_genes', type=int, default=3000)
args = parser.parse_args()

# Data
print('==> Preparing data..')

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
              size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    n_top_genes=highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def preprocess(X, nb_genes=500):
    X = np.ceil(X).astype(np.uint32)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    adata = sc.AnnData(X)
    print(adata.X.shape, adata.obs.shape, adata.var.shape)
    print(type(adata.X), type(adata.obs), type(adata.var))
    print(adata.X.dtype, np.max(adata.X), np.min(adata.X))
    adata = normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X_normalized = adata.X.astype(np.float32)
    return X_normalized, adata.raw[:, adata.var_names].X, adata.obs['size_factors']

DATA_ROOT = '../BYOL/byol-pytorch-master/datasets'

data_mat = h5py.File(f"{DATA_ROOT}/{args.dataset}/data.h5", "r")
try:
    X = np.array(data_mat['X'])
    Y = np.array(data_mat['Y'])
    unique_class = np.unique(Y)
except:
    exprs_handle = data_mat["exprs"]
    if isinstance(exprs_handle, h5py.Group):
        X = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                  exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
    else:
        X = exprs_handle[...].astype(np.float32)
    if isinstance(X, np.ndarray):
        X = np.array(X)
    else:
        X = np.array(X.toarray())
    obs = pd.DataFrame(dict_from_group(data_mat["obs"]), index=decode(data_mat["obs_names"][...]))
    Y = np.array(obs["cell_type1"])
    unique_class = np.unique(Y)
    class_dict = {}
    i = 0
    for cls in unique_class:
        class_dict.update({cls: i})
        i = i + 1
    for i, label in enumerate(Y):
        Y[i] = class_dict[label]


X, _, _ = preprocess(X, nb_genes=args.nb_genes)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X = torch.tensor(X)
Y = torch.tensor(Y.astype(np.float32))
num_cluster = len(unique_class)


class scRNADataset(torch.utils.data.Dataset):
    # new dataset class that allows to get the sample indices of mini-batch
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        data, target = self.x[index], self.y[index]
        return data, target, index

    def __len__(self):
        return len(self.x)


trainset = scRNADataset(X, Y)
loader = torch.utils.data.DataLoader(trainset, batch_size=X.shape[0], shuffle=False, num_workers=2)

if not os.path.exists(f'./dataset/{args.nb_genes}/{args.dataset}'):
    os.makedirs(f'./dataset/{args.nb_genes}/{args.dataset}')

num_data = [10]
with torch.no_grad():
    for j, data_t in enumerate(loader, 0):
        dist_list = [[] for i in range(len(num_data))]
        # get all inputs
        inputs_t, labels_t, _ = data_t
        if use_cuda:
            inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
        for i in range(len(inputs_t)):
            if i%1000 == 0:
                print(i)
            aa = torch.mul(inputs_t - inputs_t[i],inputs_t - inputs_t[i])
            dist = torch.sqrt(torch.sum(aa,dim=(1)))
            dist_m = dist[:]
            dist_m[i] = 100000
            sorted_dist = np.sort(dist_m.cpu().numpy())
            for jj in range(len(num_data)):
                dist_list[jj].append(sorted_dist[num_data[jj]])

    for ii in range(len(num_data)):
        np.savetxt(f'./dataset/{args.nb_genes}/{args.dataset}/' + str(num_data[ii]) + 'th_neighbor.txt',
                   np.array(dist_list[ii]))
