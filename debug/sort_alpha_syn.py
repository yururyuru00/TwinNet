import sys, os
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
from deeprobust.graph.data import Dataset

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph, from_scipy_sparse_matrix
from torch_geometric.datasets import KarateClub, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!" 
        self.require_mask = require_mask

        # require_lcc is False
        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels
    
    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test() # do here

    def to_torch_tensor(self, device):
        self.x = torch.from_numpy(self.features.todense()).to(device)
        self.edge_index = from_scipy_sparse_matrix(self.adj)[0].to(device)
        self.y = torch.from_numpy(self.labels).to(torch.int64).to(device)
        self.train_mask = torch.from_numpy(self.train_mask).to(device)
        self.val_mask = torch.from_numpy(self.val_mask).to(device)
        self.test_mask = torch.from_numpy(self.test_mask).to(device)
        self.num_nodes = self.x.size()[0]
        del self.adj, self.idx_train, self.idx_val, self.idx_test, \
            self.labels, self.y_train, self.y_val, self.y_test

def calc_class_dist(data, center_node, max_depth):
    self_label = data.y[center_node]
    for l in range(1, max_depth+1):
        subgraph = k_hop_subgraph(node_idx   = center_node,
                                  num_hops   = l,
                                  edge_index = data.edge_index)
        sub_nodes, _, _, _ = subgraph
        sub_node_labels = data.y[sub_nodes]
        num_diff_labels = torch.where(sub_node_labels != self_label)[0].size()[0]
        if num_diff_labels > 0:
            return l

def calc_khop_homophily(data, center_node, k_hop):
    subgraph = k_hop_subgraph(node_idx   = center_node,
                              num_hops   = k_hop,
                              edge_index = data.edge_index)
    sub_nodes, _, _, _ = subgraph
    num_neighbors = sub_nodes.size()[0]
    self_label = data.y[center_node]
    sub_node_labels = data.y[sub_nodes]
    num_same_labels = torch.where(sub_node_labels == self_label)[0].size()[0]
    return num_same_labels, num_neighbors


# load data and pre_transform
homophily = 0.0
split_seed = 1
num_layer = 4
num_hops  = 2

os.chdir('../')
path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = CustomDataset(
            root=path+'/data/syn-cora', name="h{}0-r{}".format(homophily, split_seed),
            setting="gcn", seed=15, require_mask=True
       )
data.to_torch_tensor(device)
alphas = np.load('./debug/alpha/syn_cora_alpha_twin_L{}.npy'.format(num_layer))

homophilies = []
for v in tqdm(range(data.num_nodes)):
    num_same_labels, num_neighbors = calc_khop_homophily(data, v, num_hops)
    num_same_labels_, num_neighbors_ = calc_khop_homophily(data, v, num_hops-1)
    just_k_hop_homophily = (num_same_labels - num_same_labels_) \
                           / (num_neighbors - num_neighbors_)
    homophilies.append(just_k_hop_homophily)
homophilies = np.array(homophilies)
homophily_rank = np.argsort(homophilies)

test_mask = np.load('./debug/correct_idxes/syn_cora/test_mask.npy')
test_mask_sorted = test_mask[homophily_rank]
alphas_sorted = alphas[homophily_rank]
alphas_test = alphas[test_mask]
alphas_test_sorted = alphas[test_mask_sorted]

# np.save('./{}_homophily_rank.npy'.format(dataset_name), homophily_rank)

plt.figure()
sns.heatmap(alphas, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./debug/visualize_alpha/syn_cora/alpha_not_sorted.png')
plt.close('all')

plt.figure()
sns.heatmap(alphas_sorted, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./debug/visualize_alpha/syn_cora/alpha_sorted.png')
plt.close('all')

plt.figure()
sns.heatmap(alphas_test, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./debug/visualize_alpha/syn_cora/alpha_test_not_sorted.png')
plt.close('all')

plt.figure()
sns.heatmap(alphas_test_sorted, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./debug/visualize_alpha/syn_cora/alpha_test_sorted.png')
plt.close('all')

