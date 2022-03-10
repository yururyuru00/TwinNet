import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv

mask0 = np.load('./debug/correct_idxes/syn_cora/test_mask_0.npy')
mask1 = np.load('./debug/correct_idxes/syn_cora/test_mask_5.npy')
pass