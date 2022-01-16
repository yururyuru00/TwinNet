import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GNNConv, SkipConnection, Summarize


class TwinGCN(nn.Module):
    def __init__(self, cfg):
        super(TwinGCN, self).__init__()
        self.dropout = cfg.dropout
        self.act = eval(f'nn.' + cfg.activation + '()') # ReLU or Identity

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv(cfg.task, 'gcn_conv', cfg.n_feat, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv(cfg.task, 'gcn_conv', cfg.n_hid, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))

        self.summarize = Summarize(cfg.n_hid, cfg.scope, cfg.temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)


    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for conv in self.convs:
            x, x_ = conv([x, x_], edge_index)
            x, x_ = self.act(x), self.act(x_)
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_)  # hs and hs_ is [h^1,h^2,...,h^L], each h^l is (n, d).
        return self.out_lin(h)




class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.dropout = cfg.dropout
        self.act = nn.ReLU()

        self.in_conv = GNNConv(cfg.task, 'gcn_conv', cfg.n_feat, cfg.n_hid, cfg.norm)
        
        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            self.mid_convs.append(GNNConv(cfg.task, 'gcn_conv', in_channels, cfg.n_hid, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid))
        
        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv(cfg.task, 'gcn_conv', in_channels, cfg.n_class, norm='None')

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = mid_conv(x, edge_index)
            h = self.act(h)
            x = skip(h, x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.out_conv(x, edge_index)
        
        return x



def return_net(cfg):
    # our algorithm
    if cfg.skip_connection == 'summarize': 
        if cfg.base_gnn == 'GCN':
            return TwinGCN(cfg)
        elif cfg.base_gnn == 'GAT':
            raise NotImplementedError()
        elif cfg.base_gnn == 'SAGE':
            raise NotImplementedError()

    # existing algorithms (vanilla, res, dense, or highway skip-connection)
    else:
        if cfg.base_gnn == 'GCN':
            return GCN(cfg)
        elif cfg.base_gnn == 'GAT':
            raise NotImplementedError()
        elif cfg.base_gnn == 'SAGE':
            raise NotImplementedError()