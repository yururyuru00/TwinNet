import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


def calc_homophily_of_all_nodes(target, y, edge_index, max_depth, mode='single'):
    one_hop_neighbors = {}
    all_nodes = range(y.size(0))
    for vi in all_nodes:
        ids = torch.where(edge_index[0]==vi)[0]
        neighbors = edge_index[1][ids].tolist()
        one_hop_neighbors[vi] = neighbors

    all_vertex_scores = []
    for vi in tqdm(target):
        khop_neighbors_of_vi = [set() for _ in range(max_depth+1)]
        dfs_khop_neighbors(vi, 0, max_depth, khop_neighbors_of_vi, one_hop_neighbors)
        all_vertex_scores.append(calc_homophily_of_vi(vi, y, khop_neighbors_of_vi, mode))
    all_vertex_scores = torch.stack(all_vertex_scores, dim=0)
    return all_vertex_scores


def dfs_khop_neighbors(node, depth, max_depth, khop_neighbors, one_hop_neighbors):
    if node in khop_neighbors[depth]:
        return
    else:
        khop_neighbors[depth].add(node)
    
    if depth+1 > max_depth:
        return
    for n in one_hop_neighbors[node]:
        dfs_khop_neighbors(n, depth+1, max_depth, khop_neighbors, one_hop_neighbors)


def calc_homophily_of_vi(vi, y, khop_neighbors_of_vi, mode):
    n_layer = len(khop_neighbors_of_vi)
    if mode == 'multi':
        n_class = data.y.size(1)
    homophily_score_layerwise = []
    for l in range(1, n_layer):
        neighbor_size = len(khop_neighbors_of_vi[l])
        if neighbor_size == 0:
            homophily_score_layerwise.append(0.)
        else:
            ids_neighbors = torch.tensor([[n for n in khop_neighbors_of_vi[l]]])
            self_labels = y[vi]
            labels_neighbors = y[ids_neighbors]
            if mode == 'single':
                acc = len(torch.where(labels_neighbors == self_labels)[0].tolist()) / neighbor_size
            else:
                acc = 0.
                for label_neighbor in labels_neighbors:
                    acc += torch.where(self_labels*label_neighbor==1)[0].size(0)
                acc /= (n_class * neighbor_size)
            homophily_score_layerwise.append(acc)
    return torch.tensor(homophily_score_layerwise)


#---------------------------------------------------------------------
data_name = 'PubMed'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '../data/{}'.format(data_name).lower()


if data_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root          = root,
                        name          = data_name,
                        split         = 'full',
                        transform     = T.AddSelfLoops())
    data = dataset[0].to(device)
    max_depth = 9

    y = data.y
    edge_index = data.edge_index
    target = torch.where(data.test_mask==True)[0].tolist() # only test nodes
    all_nodes_scores = calc_homophily_of_all_nodes(target, y, edge_index, max_depth)
    np.save('./alpha/PubMed_test_true.npy', all_nodes_scores.to('cpu').detach().numpy().copy())


else: # data_name == 'PPI'
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    max_depth = 6
    loader_names = ['train_loader', 'val_loader', 'test_loader']
    for loader_name in loader_names:
        print(loader_name + 'start')
        loader = eval(loader_name)
        for graph_id, data in tqdm(enumerate(loader)):
            y = data.y
            edge_index = data.edge_index
            target = range(y.size(0))
            all_nodes_scores = calc_homophily_of_all_nodes(target, y, edge_index, max_depth, mode='multi')
            
            graph_name = '{}_g{}'.format(loader_name, graph_id)
            np.save('./result/homophily_score/PPI/{}_homo_score.npy'.format(graph_name), 
                    all_nodes_scores.to('cpu').detach().numpy().copy())
    