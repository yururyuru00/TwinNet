import numpy as np
import torch
from torch_geometric.datasets import MixHopSyntheticDataset
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

from utils import ExtractSubGraph


homophily = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MixHopSyntheticDataset(
            root          = './data/syn_mini/h{}'.format(homophily),
            pre_transform = ExtractSubGraph([1,2,3]),
            homophily     = homophily
          )
data = dataset[0].to(device)

target_label_ids = [2,7,5]
target_nodes = []
for label_id in target_label_ids:
    sub_nodes = torch.where(data.y == label_id)[0].tolist()
    target_nodes += sub_nodes
target_edge_index, _ = subgraph(subset     = target_nodes,
                                edge_index = data.edge_index)
sub_data = Data(data.x[sub_nodes], target_edge_index, data.y[sub_nodes])
sub_data.train_mask = torch.full()

print(data)

