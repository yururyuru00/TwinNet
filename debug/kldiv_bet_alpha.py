import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])

def make_binomial(n, p=0.5):
    dist = np.array( [(math.factorial(n) / (math.factorial(i) * math.factorial(n-i))) * np.power(p, i) * np.power(1-p, n-i) \
                       for i in range(1, n+1)] )
    return dist * sum(dist)

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


dataset_name = 'PubMed'
num_layer = 9
temparature= 1

alphas = {}
alphas['twin'] = np.load('./alpha/{}_alpha_L{}_t{}.npy'.format(dataset_name, num_layer, temparature))
try:
    alphas['jk'] = np.load('./alpha/{}_alpha_jk_L{}.npy'.format(dataset_name, num_layer))
except FileNotFoundError: # it means jknet failed because of out of memory
    pass
n_nodes = alphas['twin'].shape[0]
binomial = make_binomial(num_layer)
# alphas['skip'] = np.repeat(binomial.reshape(1, -1), n_nodes, axis=0)
alphas['gnn']  = np.repeat(np.identity(num_layer)[-1].reshape(1, -1), n_nodes, axis=0)



# make alpha_true based on 100% class label
dataset = Planetoid(root      = '../data/'+dataset_name,
                    name      = dataset_name,
                    transform = T.AddSelfLoops())
data = dataset[0]
alpha_true    = np.zeros((data.num_nodes, num_layer))
homophilies = []
for v in tqdm(range(data.num_nodes)):
    for l in range(num_layer):
        homophily = calc_khop_homophily(data, v, k_hop=l+1)
        alpha_true[v][l] = homophily

# normalize alpha_true
epsilon_ary = np.full_like(alpha_true, 1e-5)
alpha_true += epsilon_ary
rowsum = np.sum(alpha_true, axis=-1)
rowsum_inv = np.power(rowsum, -1)
alpha_true = np.array([np.dot(vec, normalize_coefficient)
                    for vec, normalize_coefficient in zip(alpha_true, rowsum_inv)])
np.save('./alpha/true/{}'.format(dataset_name), alpha_true)

epsilon_vec = np.array([1e-5 for _ in range(num_layer)])
kl_divs = {}
for model_name, alpha in alphas.items():
    kl_divs[model_name] = []
    for v_alpha_true, v_alpha in zip(alpha, alpha_true):
        v_alpha_true += epsilon_vec
        v_alpha += epsilon_vec
        kl_div = calc_kldiv(v_alpha_true, v_alpha)
        kl_divs[model_name].append(kl_div)


divs  = [v for n, v in kl_divs.items()]
names = [n for n, v in kl_divs.items()]

fig, ax = plt.subplots()
ax.boxplot(divs)
ax.set_xticklabels(names)
# plt.ylim([0,2])
plt.show()
plt.savefig('.result.png')