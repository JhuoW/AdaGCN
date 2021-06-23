import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os,sys
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import copy

class SparseDropout(nn.Module):
    def __init__(self, p:float = 0.5):
        super(SparseDropout, self).__init__()
        self.p = p
    def forward(self, x: torch.sparse.FloatTensor):
        shape = x.shape
        x = x.coalesce()
        drop_val = F.dropout(x._values(), self.p, self.training)
        return torch.sparse.FloatTensor(x._indices(), drop_val, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def aug_normalized_adjacency(adj):
    """ A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2 """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalized_adjacency(adj): 
    """ A' = (D)^-1/2 * ( A) * (D)^-1/2 """ 
    adj = adj
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = torch.LongTensor(np.argmax(labels, axis=1))
    n_classes = max(labels) + 1
    n_nodes, nfeat = features.shape
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    train_mask = torch.from_numpy(sample_mask(idx_train, n_nodes))
    val_mask = torch.from_numpy(sample_mask(idx_val, n_nodes))
    test_mask = torch.from_numpy(sample_mask(idx_test, n_nodes))
    features = torch.Tensor(normalize_features(features).todense())

    ori_adj = sp.csr_matrix(adj)

    aug_adj = aug_normalized_adjacency(ori_adj)

    ori_adj = sparse_mx_to_torch_sparse_tensor(ori_adj)
    aug_adj = sparse_mx_to_torch_sparse_tensor(aug_adj)
    return ori_adj, aug_adj, features, labels, train_mask, val_mask, test_mask, n_nodes, nfeat, n_classes.numpy()


def exclude_idx(idx: np.ndarray, idx_exclude_list) -> np.ndarray:
    
    idx_exclude = np.concatenate(idx_exclude_list)


    return np.array([i for i in idx if i not in idx_exclude]) 



def train_stopping_split( idx: np.ndarray, 
                          labels: np.ndarray, 
                          ntrain_per_class: int = 20,   
                          nstopping: int = 500, 
                          seed: int = 2413340114) -> Tuple[np.ndarray, np.ndarray]:


    rnd_state = np.random.RandomState(seed)
    train_idx_split = [] 
    for i in range(max(labels) + 1): 
        train_idx_split.append(rnd_state.choice(    
                idx[labels == i], ntrain_per_class, replace=False))


    train_idx = np.concatenate(train_idx_split)
    
    # 500个验证集节点
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx

def known_unknown_split(
        idx: np.ndarray, 
        nknown: int = 1500, 
        seed: int = 4143496719) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)  
    unknown_idx = exclude_idx(idx, [known_idx])  
    return known_idx, unknown_idx

def gen_splits(
        args, labels):
    """
    standard setting
    PPNP: 训练集每个类别20个节点，20*7 = 140,  500个验证集节点，  
    """

    all_idx = np.arange(labels.shape[0])  
    
    train_idx, stopping_idx = train_stopping_split(all_idx, labels, args.n_train_per_class, args.n_val)   
    test_idx = exclude_idx(all_idx, [train_idx, stopping_idx])
    return torch.LongTensor(train_idx), torch.LongTensor(stopping_idx), torch.LongTensor(test_idx)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
