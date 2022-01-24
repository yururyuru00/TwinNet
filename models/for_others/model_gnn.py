import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer import GNNConv, SkipConnection


class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('gcn_conv', cfg.n_feat, cfg.n_hid, cfg.norm)

        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense (h = h || x)
                in_channels = cfg.n_hid * l
            self.mid_convs.append(GNNConv('gcn_conv', in_channels, cfg.n_hid, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels, in_channels))
        
        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('gcn_conv', in_channels, cfg.n_class, norm='None')

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = mid_conv(x, edge_index)
            x = skip((h, x))
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.out_conv(x, edge_index)
        return x


# SAGE do not use skip-connection because SAGE already use skip-connection (h = AxW + xW_)
class SAGE(nn.Module):
    def __init__(self, cfg):
        super(SAGE, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm)
        self.mid_convs = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            self.mid_convs.append(GNNConv('sage_conv', cfg.n_hid, cfg.n_hid, cfg.norm))
        self.out_conv = GNNConv('sage_conv', cfg.n_hid, cfg.n_class, norm='None')

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_conv in self.mid_convs:
            x = mid_conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_conv(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, cfg):
        super(GAT, self).__init__()
        self.dropout = cfg.dropout

        self.in_conv = GNNConv('gat_conv', cfg.n_feat, cfg.n_hid, cfg.norm,
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
            mid_conv = GNNConv('gat_conv', in_channels, cfg.n_hid, cfg.norm,
                               n_heads     = [cfg.n_head, cfg.n_head],
                               iscat       = [True, True],
                               dropout_att = cfg.dropout_att)
            self.mid_convs.append(mid_conv)
            self.skips.append(SkipConnection(cfg.skip_connection, in_channels*cfg.n_head, in_channels*cfg.n_head))


        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('gat_conv', in_channels, cfg.n_class, norm='None',
                                n_heads     = [cfg.n_head, cfg.n_head_last],
                                iscat       = [True, False],
                                dropout_att = cfg.dropout_att)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.in_conv(x, edge_index)
        x = F.elu(x)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            x = F.dropout(x, self.dropout, training=self.training)
            h = mid_conv(x, edge_index)
            x = skip((h, x))
            x = F.elu(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_conv(x, edge_index)
        return x
