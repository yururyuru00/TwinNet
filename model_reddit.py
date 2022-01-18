import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import Summarize, GNNConv, SkipConnection


def conv_for_gpumemory(x_all, loader, conv, device):
    if isinstance(x_all, list): # if x_all is [x_all, x_all_], we use twin-sage
        x_all_ = x_all[1]
        x_all  = x_all[0]
        xs, xs_ = [], []
        for batch_size, n_id, adj in loader:
            edge_index, _, size = adj.to(device)
            x, x_ = x_all[n_id].to(device), x_all_[n_id].to(device)
            x_target = x[:size[1]]
            x_ = x_[:batch_size] # because x_ do not use aggregate

            x, x_ = conv([(x, x_target), x_], edge_index)
            xs.append(x)
            xs_.append(x_)
        x_all  = torch.cat(xs, dim=0)
        x_all_ = torch.cat(xs_, dim=0)
        return x_all, x_all_

    else: # if x_all is x_all, we use sigle-sage
        xs = []
        for batch_size, n_id, adj in loader:
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            x = conv((x, x_target), edge_index)
            xs.append(x)
        x_all = torch.cat(xs, dim=0)
        return x_all


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

    def forward(self, x, adjs, batch_size):
        hs, hs_ = [], []
        x_ = x.clone().detach()[:batch_size]
        for l, (edge_index, _, size) in enumerate(adjs): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, x_ = self.convs[l]([(x, x_target), x_], edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
            x, x_ = self.act(x), self.act(x_)
            x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                    F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)
        hs  = [h[:batch_size] for h in hs]

        h = self.summarize(hs, hs_) # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)

    def inference(self, x, loader, device):
        # we do not use dropout because inferense is test
        hs, hs_ = [], []
        x_ = x.clone().detach()
        for conv in self.convs:
            x, x_ = conv_for_gpumemory([x, x_], loader, conv, device)
            x, x_ = self.act(x), self.act(x_)
            hs.append(x)
            hs_.append(x_)

        h = self.summarize(hs, hs_)
        return self.out_lin(h)



class SAGE(nn.Module):
    def __init__(self, cfg):
        super(SAGE, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.in_conv = GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm)

        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            self.mid_convs.append(GNNConv('sage_conv', in_channels, cfg.n_hid, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid))

        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv('sage_conv', in_channels, cfg.n_class, norm='None')

    def forward(self, x, adjs, _):
        in_adj, mid_adjs, out_adj = adjs[0], adjs[1:-1], adjs[-1]

        x_target = x[:in_adj.size[1]] # Target nodes are always placed first.
        x = self.in_conv((x, x_target), in_adj.edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_adj, mid_conv, skip in zip(mid_adjs, self.mid_convs, self.skips): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:mid_adj.size[1]]
            h = mid_conv((x, x_target), mid_adj.edge_index)
            h = F.relu(h)
            x = skip(h, x_target)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_target = x[:out_adj.size[1]]
        x = self.out_conv((x, x_target), out_adj.edge_index)
    
        return x

    def inference(self, x, loader, device):
        # we do not use dropout because inferense is test
        x = conv_for_gpumemory(x, loader, self.in_conv, device)
        x = F.relu(x)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = conv_for_gpumemory(x, loader, mid_conv, device)
            h = F.relu(h)
            x = skip(h, x)

        x = conv_for_gpumemory(x, loader, self.out_conv, device)
        
        return x



def return_net(cfg):
    # our algorithm (summarize skip-connection)
    if cfg.skip_connection == 'summarize': 
        return TwinSAGE(cfg)

    # existing algorithms ([vanilla, res, dense, highway] skip-connection)
    else:
        return SAGE(cfg)