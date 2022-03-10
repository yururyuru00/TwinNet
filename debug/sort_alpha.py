import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import KarateClub, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


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
    return num_same_labels / num_neighbors


# load data and pre_transform
dataset_name = 'Cora'
num_layer = 7
num_hops  = 7
temparature = 1

if dataset_name == 'KarateClub':
    dataset = KarateClub(transform = T.AddSelfLoops())
elif dataset_name == 'Arxiv':
    dataset = PygNodePropPredDataset('ogbn-arxiv', '../data/'+dataset_name)
else:
    dataset = Planetoid(root      = '../data/'+dataset_name,
                        name      = dataset_name,
                        transform = T.AddSelfLoops())
data = dataset[0]
alphas = np.load('./alpha/{}_alpha_L{}_t{}_orth.npy'.format(dataset_name, num_layer, temparature))

homophilies = []
for v in tqdm(range(data.num_nodes)):
    homophily = calc_khop_homophily(data, v, num_hops)
    homophilies.append(homophily)
homophilies = np.array(homophilies)
homophily_rank = np.argsort(homophilies)

test_mask = np.load('./correct_idxes/{}/test_mask.npy'.format(dataset_name))
test_mask_sorted = test_mask[homophily_rank]
alphas_sorted = alphas[homophily_rank]
alphas_test = alphas[test_mask]
alphas_test_sorted = alphas[test_mask_sorted]

# np.save('./{}_homophily_rank.npy'.format(dataset_name), homophily_rank)

plt.figure()
sns.heatmap(alphas, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./visualize_alpha/{}/alpha_not_sorted.png'.format(dataset_name))
plt.close('all')

plt.figure()
sns.heatmap(alphas_sorted, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./visualize_alpha/{}/alpha_sorted.png'.format(dataset_name))
plt.close('all')

plt.figure()
sns.heatmap(alphas_test, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./visualize_alpha/{}/alpha_test_not_sorted.png'.format(dataset_name))
plt.close('all')

plt.figure()
sns.heatmap(alphas_test_sorted, vmin=0, vmax=1, cmap='Reds')
plt.savefig('./visualize_alpha/{}/alpha_test_sorted.png'.format(dataset_name))
plt.close('all')
