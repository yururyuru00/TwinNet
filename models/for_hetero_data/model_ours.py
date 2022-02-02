import copy

import torch
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict

from ..layer import Summarize


class TwinRGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(TwinRGCNConv, self).__init__()

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

    def forward(self, x_dict, x_dict_, adj_t_dict):
        out_dict, out_dict_ = {}, {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)
        for key_, x_ in x_dict_.items():
            out_dict_[key_] = self.root_lins[key_](x_)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x, x_ = x_dict[key[0]], x_dict_[key[2]]
            out, out_ = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean')), \
                        self.rel_lins[key_str](x_) # new repair
            out_dict[key[2]].add_(out)
            out_dict_[key[2]].add_(out_) # new repair

        return out_dict, out_dict_


class TwinRGCN(torch.nn.Module):
    def __init__(self, cfg, num_nodes_dict, x_types, edge_types):
        super(TwinRGCN, self).__init__()
        self.target_type = x_types[0]
        self.dropout = cfg.dropout

        node_types = list(num_nodes_dict.keys())
        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], cfg.n_feat))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            TwinRGCNConv(cfg.n_feat, cfg.n_hid, node_types, edge_types))
        for _ in range(cfg.n_layer - 1):
            self.convs.append(
                TwinRGCNConv(cfg.n_hid, cfg.n_hid, node_types, edge_types))
        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel,
                                   channels    = cfg.n_hid, 
                                   temparature = cfg.temparature)
        self.out_lin = Linear(cfg.n_hid, cfg.n_class)

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

        hs, hs_ = [], []
        x_dict_ = copy.deepcopy(x_dict)
        for conv in self.convs:
            x_dict, x_dict_ = conv(x_dict, x_dict_, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout,
                                        training=self.training)
            for key_, x_ in x_dict_.items():
                x_dict_[key_] = F.relu(x_)
                x_dict_[key_] = F.dropout(x_, p=self.dropout,
                                          training=self.training)
            hs.append(x_dict)
            hs_.append(x_dict_)

        hs  = [h[self.target_type] for h in hs]
        hs_ = [h_[self.target_type] for h_ in hs_]
        h, alpha = self.summarize(hs, hs_)
        return self.out_lin(h), alpha