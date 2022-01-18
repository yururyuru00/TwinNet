import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM
from torch.nn import LayerNorm, BatchNorm1d, Identity

from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree


class GNNConv(nn.Module):
    def __init__(self, conv_name, in_channels, out_channels, norm,
                 self_loop=True, n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()

        if conv_name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels, add_self_loops=self_loop)
            self.conv_ = self.conv.lin

        elif conv_name == 'sage_conv':
            self.conv  = SAGEConv(in_channels, out_channels, root_weight=self_loop)
            self.conv_root     = self.conv.lin_r
            self.conv_neighbor = self.conv.lin_l
            self.conv_ = lambda x_: self.conv_root(x_) + self.conv_neighbor(x_)

        elif conv_name == 'gat_conv':
            if iscat[0]: # if previous gatconv's cat is True
                in_channels = in_channels * n_heads[0]
            self.conv  = GATConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 heads=n_heads[1],
                                 concat=iscat[1],
                                 dropout=dropout_att,
                                 add_self_loops=self_loop)
            self.conv_ = self.conv.lin_src
            if iscat[1]: # if this gatconv's cat is True
                out_channels = out_channels * n_heads[1]

        if norm == 'LayerNorm':
            self.norm, self.norm_ = LayerNorm(out_channels), LayerNorm(out_channels)
        elif norm == 'BatchNorm1d':
            self.norm, self.norm_ = BatchNorm1d(out_channels), BatchNorm1d(out_channels)
        else:
            self.norm, self.norm_ = Identity(), Identity()


    def forward(self, xs, edge_index):
        if isinstance(xs, list): # if xs is [x, x_], we use twin-gnn
            x = self.conv(xs[0], edge_index)
            x_ = self.conv_(xs[1])
            return self.norm(x), self.norm_(x_)

        else: # if xs is x, we use sigle-gnn
            x = xs
            x = self.conv(x, edge_index)
            return self.norm(x)


# if cfg.skip_connection is 'summarize'
class Summarize(nn.Module):
    def __init__(self, scope, kernel, channels, temparature):
        super(Summarize, self).__init__()
        self.scope = scope
        self.kernel = kernel
        self.att_temparature = temparature
        
        self.att = Linear(2 * channels, 1)
        self.att.reset_parameters()
        self.weight = nn.Parameter(torch.ones(channels), requires_grad=True)

    def forward(self, hs, hs_):
        h = torch.stack(hs, dim=1)  # h is (n, L, d).
        h_ = torch.stack(hs_, dim=1)  # h_ is also (n, L, d).

        if self.scope == 'local':
            query, key = h_, h
        else: # if 'global'
            n_nodes = h.size()[0]
            query = h
            key = torch.mean(h_, dim=0, keepdim=True).repeat((n_nodes, 1, 1))

        if self.kernel == 'dp':
            alpha = (query * key).sum(dim=-1)
        
        elif self.kernel == 'sdp':
            alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])
        
        elif self.kernel == 'wdp':
            alpha = (query * key * self.weight).sum(dim=-1) / math.sqrt(query.size()[-1])

        elif self.kernel == 'ad':
            query_key = torch.cat([query, key], dim=-1)
            alpha = self.att(query_key).squeeze(-1)
            
        elif self.kernel == 'mx': # mix of dp and ad 
            query_key = torch.cat([query, key], dim=-1)
            alpha_ad = self.att(query_key).squeeze(-1)
            alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

        alpha_softmax = torch.softmax(alpha/self.att_temparature, dim=-1)
        return (h * alpha_softmax.unsqueeze(-1)).sum(dim=1)



# if cfg.skip_connection is 'vanilla', 'res', 'dense', or 'highway'
class SkipConnection(nn.Module):
    def __init__(self, skip_connection, n_hidden):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection
        if self.skip_connection == 'highway':
            self.linear = Linear(n_hidden, n_hidden)

    def forward(self, h, x):
        if self.skip_connection == 'vanilla':
            return h

        elif self.skip_connection == 'res':
            return h + x # maybe h*0.5 + x*0.5

        elif self.skip_connection == 'dense':
            return torch.cat([h, x], dim=-1)

        elif self.skip_connection == 'highway':
            gating_weights = torch.sigmoid(self.linear(x))
            ones = torch.ones_like(gating_weights)
            return h*gating_weights + x*(ones-gating_weights) # h*W + x*(1-W)
    