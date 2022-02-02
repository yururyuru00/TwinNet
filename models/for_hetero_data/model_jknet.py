import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict

from ..layer import JumpingKnowledge


class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.rel_lins = ModuleDict({
            f'{key[0]}_{key[1]}_{key[2]}': Linear(in_channels, out_channels,
                                                  bias=False)
            for key in edge_types
        })

        self.root_lins = ModuleDict({
            key: Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[key[2]].add_(out)

        return out_dict


class JKRGCN(torch.nn.Module):
    def __init__(self, cfg, num_nodes_dict, x_types, edge_types):
        super(JKRGCN, self).__init__()
        self.target_type = x_types[0]
        self.dropout = cfg.dropout

        node_types = list(num_nodes_dict.keys())
        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], cfg.n_feat))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            RGCNConv(cfg.n_feat, cfg.n_hid, node_types, edge_types))
        for _ in range(cfg.n_layer - 1):
            self.convs.append(
                RGCNConv(cfg.n_hid, cfg.n_hid, node_types, edge_types))

        self.jk = JumpingKnowledge(mode       = cfg.jk_mode, 
                                   channels   = cfg.n_hid,
                                   num_layers = cfg.n_layer)
        if cfg.jk_mode == 'cat':
            self.out_lin = nn.Linear(cfg.n_hid*cfg.n_layer, cfg.n_class)
        else: # if jk_mode == 'max' or 'lstm'
            self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb

        hs = []
        for conv in self.convs:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout,
                                        training=self.training)
            hs.append(x_dict)

        hs = [h[self.target_type] for h in hs]
        h, alpha = self.jk(hs)
        return self.out_lin(h), alpha