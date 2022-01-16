import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchviz import make_dot
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from layer import GNNConv, Summarize, SkipConnection


class TwinGCN(nn.Module):

    def __init__(self, cfg):
        super(TwinGCN, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv('gcn_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('gcn_conv', cfg.n_hid, cfg.n_hid, cfg.self_loop, cfg.norm))

        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel, 
                                   channels    = cfg.n_hid, 
                                   temparature = cfg.temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for l, conv in enumerate(self.convs):
            x, x_ = conv(x, x_, edge_index)
            x, x_ = F.relu(x), F.relu(x_)
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_)  # hs and hs_ is [h^1,h^2,...,h^L], each h^l is (n, d).
        return self.out_lin(h)


class TwinSAGE(nn.Module):
    def __init__(self, cfg):
        super(TwinSAGE, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('sage_conv', cfg.n_hid, cfg.n_hid, cfg.self_loop, cfg.norm))

        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel, 
                                   channels    = cfg.n_hid, 
                                   temparature = cfg.temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for l, conv in enumerate(self.convs):
            x, x_ = conv(x, x_, edge_index)
            x, x_ = F.relu(x), F.relu(x_)
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_) # xs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)


class TwinGAT(nn.Module):

    def __init__(self, cfg):
        super(TwinGAT, self).__init__()
        self.dropout = cfg.dropout
    
        self.convs = torch.nn.ModuleList()
        in_conv = GNNConv('gat_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm,
                          n_heads     = [1, cfg.n_head],
                          iscat       = [False, True],
                          dropout_att = cfg.dropout_att)
        self.convs.append(in_conv)
        for _ in range(1, cfg.n_layer):
            conv = GNNConv('gat_conv', cfg.n_hid, cfg.n_hid, cfg.self_loop, cfg.norm,
                           n_heads     = [cfg.n_head, cfg.n_head],
                           iscat       = [True, True],
                           dropout_att = cfg.dropout_att)
            self.convs.append(conv)

        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel, 
                                   channels    = cfg.n_hid, 
                                   temparature = cfg.temparature)
        self.out_lin = nn.Linear(cfg.n_hid * cfg.n_head, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for l, conv in enumerate(self.convs):
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            x, x_ = conv(x, x_, edge_index)
            x, x_ = F.elu(x), F.elu(x_)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_)  # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)


class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('gcn_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm)
        
        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            self.mid_convs.append(GNNConv('gcn_conv', in_channels, cfg.n_hid, cfg.self_loop, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid))
        
        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('gcn_conv', in_channels, cfg.n_class, cfg.self_loop, norm='None')

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = mid_conv(x, edge_index)
            h = F.relu(h)
            x = skip(h, x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.out_conv(x, edge_index)
        
        return x, None


class SAGE(nn.Module):
    def __init__(self, cfg):
        super(SAGE, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm)

        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            self.mid_convs.append(GNNConv('sage_conv', in_channels, cfg.n_hid, cfg.self_loop, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid))

        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('sage_conv', in_channels, cfg.n_class, cfg.self_loop, norm='None')

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = mid_conv(x, edge_index)
            h = F.relu(h)
            x = skip(h, x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_conv(x, edge_index)

        return x, None


class GAT(nn.Module):
    def __init__(self, cfg):
        super(GAT, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('gat_conv', cfg.n_feat, cfg.n_hid, cfg.self_loop, cfg.norm,
                               n_heads     = [1, cfg.n_head],
                               iscat       = [False, True],
                               dropout_att = cfg.dropout_att)
        
        self.mid_convs = torch.nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            mid_conv = GNNConv('gat_conv', in_channels, cfg.n_hid, cfg.self_loop, cfg.norm,
                               n_heads     = [cfg.n_head, cfg.n_head],
                               iscat       = [True, True],
                               dropout_att = cfg.dropout_att)
            self.mid_convs.append(mid_conv)
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid*cfg.n_head))

        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('gat_conv', in_channels, cfg.n_class, cfg.self_loop, norm='None',
                                n_heads     = [cfg.n_head, cfg.n_head_last],
                                iscat       = [True, False],
                                dropout_att = cfg.dropout_att)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.in_conv(x, edge_index)
        x = F.elu(x)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = F.dropout(x, self.dropout, training=self.training)
            h = mid_conv(h, edge_index)
            h = F.elu(h)
            x = skip(h, x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_conv(x, edge_index)
        
        return x, None



def return_net(cfg):
    # our algorithm (attention skip-connection)
    if cfg.skip_connection == 'summarize': 
        if cfg.base_gnn == 'GCN':
            return TwinGCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return TwinSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return TwinGAT(cfg)

    # existing algorithms (vanilla, res, dense, or highway skip-connection)
    else:
        if cfg.base_gnn == 'GCN':
            return GCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return SAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return GAT(cfg)