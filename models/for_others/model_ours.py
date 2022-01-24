import torch.nn as nn
import torch.nn.functional as F

from ..layer import GNNConv, Summarize, SkipConnection


class TwinGCN(nn.Module):
    def __init__(self, cfg):
        super(TwinGCN, self).__init__()
        self.dropout = cfg.dropout
        self.act = eval(f'nn.' + cfg.activation + '()') # ReLU or Identity

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.convs.append(GNNConv('gcn_conv', cfg.n_feat, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))
        self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_feat, cfg.n_hid))
        
        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense ( h = [h || x] )
            in_channels = cfg.n_hid + cfg.n_hid
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('gcn_conv', in_channels, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels, cfg.n_hid))
        
        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel, 
                                   channels    = in_channels, 
                                   temparature = cfg.temparature)
        self.out_lin = nn.Linear(in_channels, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for conv, skip in zip(self.convs, self.skips):
            h, h_ = conv([x, x_], edge_index) # twin-Conv
            x, x_ = skip([(h, x), (h_, x_)]) # twin-Skip
            x, x_ = self.act(x), self.act(x_)
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_)  # hs and hs_ is [h^1,h^2,...,h^L], each h^l is (n, d). # if dense, (n, 2d)
        return self.out_lin(h)


# TwinSAGE do not use skip-connection because SAGE already use skip-connection (h = AxW + xW_)
class TwinSAGE(nn.Module):
    def __init__(self, cfg):
        super(TwinSAGE, self).__init__()
        self.dropout = cfg.dropout
        self.act = eval(f'nn.' + cfg.activation + '()') # ReLU or Identity

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv('sage_conv', cfg.n_hid, cfg.n_hid, cfg.norm, self_loop=cfg.self_loop))

        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel, 
                                   channels    = cfg.n_hid, 
                                   temparature = cfg.temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for l, conv in enumerate(self.convs):
            x, x_ = conv([x, x_], edge_index) # twin-Conv
            x, x_ = self.act(x), self.act(x_)
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
        self.act = eval(f'nn.' + cfg.activation + '()') # ELU or Identity
    
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        in_conv = GNNConv('gat_conv', cfg.n_feat, cfg.n_hid, cfg.norm,
                          self_loop   = cfg.self_loop,
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
                           self_loop   = cfg.self_loop,
                           n_heads     = [cfg.n_head, cfg.n_head],
                           iscat       = [True, True],
                           dropout_att = cfg.dropout_att)
            self.convs.append(conv)
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels*cfg.n_head, cfg.n_hid*cfg.n_head))

        self.summarize = Summarize(scope       = cfg.scope,
                                   kernel      = cfg.kernel,
                                   channels    = in_channels*cfg.n_head,
                                   temparature = cfg.temparature)
        self.out_lin   = nn.Linear(in_channels*cfg.n_head, cfg.n_class)

    def forward(self, x, edge_index):
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for conv, skip in zip(self.convs, self.skips):
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            h, h_ = conv([x, x_], edge_index) # twin-Conv
            x, x_ = skip([(h, x), (h_, x_)]) # twin-Skip
            x, x_ = self.act(x), self.act(x_)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_) # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)
