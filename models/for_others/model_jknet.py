import torch.nn as nn
import torch.nn.functional as F

from ..layer import GNNConv, SkipConnection
from torch_geometric.nn import JumpingKnowledge


class JKGCN(nn.Module):
    def __init__(self, cfg):
        super(JKGCN, self).__init__()
        self.dropout = cfg.dropout

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.convs.append(GNNConv('gcn_conv', cfg.n_feat, cfg.n_hid, cfg.norm))
        self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_feat, cfg.n_hid))
        
        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense ( h = [h || x] )
            in_channels = cfg.n_hid + cfg.n_hid
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('gcn_conv', in_channels, cfg.n_hid, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels, cfg.n_hid))
        
        self.jk = JumpingKnowledge(mode       = cfg.jk_mode, 
                                   channels   = in_channels,
                                   num_layers = cfg.n_layer)
        if cfg.jk_mode == 'cat':
            self.out_lin = nn.Linear(in_channels*cfg.n_layer, cfg.n_class)
        else: # if jk_mode == 'max' or 'lstm'
            self.out_lin = nn.Linear(in_channels, cfg.n_class)

    def forward(self, x, edge_index):
        hs = []
        for conv, skip in zip(self.convs, self.skips):
            h = conv(x, edge_index) # twin-Conv
            x = skip((h, x)) # twin-Skip
            x = F.relu(x)
            x = F.dropout(x,  self.dropout, training=self.training)
            hs.append(x)

        h = self.jk(hs)  # hs and hs_ is [h^1,h^2,...,h^L], each h^l is (n, d). # if dense, (n, 2d)
        return self.out_lin(h)


# JKSAGE do not use skip-connection because SAGE already use skip-connection (h = AxW + xW_)
class JKSAGE(nn.Module):
    def __init__(self, cfg):
        super(JKSAGE, self).__init__()
        self.dropout = cfg.dropout

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('sage_conv', cfg.n_hid, cfg.n_hid, cfg.norm))

        self.jk = JumpingKnowledge(mode       = cfg.jk_mode, 
                                   channels   = cfg.n_hid,
                                   num_layers = cfg.n_layer)
        if cfg.jk_mode == 'cat':
            self.out_lin = nn.Linear(cfg.n_hid*cfg.n_layer, cfg.n_class)
        else: # if jk_mode == 'max' or 'lstm'
            self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs = []
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_index) # twin-Conv
            x = F.relu(x)
            x = F.dropout(x,  self.dropout, training=self.training)
            hs.append(x)

        h = self.jk(hs) # xs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)


class JKGAT(nn.Module):
    def __init__(self, cfg):
        super(JKGAT, self).__init__()
        self.dropout = cfg.dropout
    
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        in_conv = GNNConv('gat_conv', cfg.n_feat, cfg.n_hid, cfg.norm,
                          n_heads     = [1, cfg.n_head],
                          iscat       = [False, True],
                          dropout_att = cfg.dropout_att)
        self.convs.append(in_conv)
        self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_feat, cfg.n_hid*cfg.n_head))

        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense (h = h || x)
            in_channels = cfg.n_hid + cfg.n_hid
        for _ in range(1, cfg.n_layer):
            conv = GNNConv('gat_conv', in_channels, cfg.n_hid, cfg.norm,
                           n_heads     = [cfg.n_head, cfg.n_head],
                           iscat       = [True, True],
                           dropout_att = cfg.dropout_att)
            self.convs.append(conv)
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels*cfg.n_head, cfg.n_hid*cfg.n_head))

        self.jk = JumpingKnowledge(mode       = cfg.jk_mode, 
                                   channels   = cfg.n_hid,
                                   num_layers = cfg.n_layer)
        if cfg.jk_mode == 'cat':
            self.out_lin = nn.Linear(cfg.n_hid*cfg.n_layer, cfg.n_class)
        else: # if jk_mode == 'max' or 'lstm'
            self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs = []
        for conv, skip in zip(self.convs, self.skips):
            x = F.dropout(x,  self.dropout, training=self.training)
            h = conv(x, edge_index) # twin-Conv
            x = skip((h, x)) # twin-Skip
            x = F.elu(x)
            hs.append(x)

        h = self.jk(hs) # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)
