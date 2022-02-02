from hydra import utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer import conv_for_gpumemory, Summarize, GNNConv


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


    def forward(self, x, adjs, batch_size=None):
        hs, hs_ = [], []
        x_ = x.clone().detach()[:batch_size]
        if isinstance(adjs, list):
            for l, (edge_index, _, size) in enumerate(adjs): # size is [B_l's size, B_(l+1)'s size]
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x, x_    = self.convs[l]([(x, x_target), x_], edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
                x, x_    = self.act(x), self.act(x_)
                x, x_    = F.dropout(x,  self.dropout, training=self.training), \
                           F.dropout(x_, self.dropout, training=self.training)
                hs.append(x)
                hs_.append(x_)
            hs  = [h[:batch_size] for h in hs]
        else:
            edge_index = adjs
            for conv in self.convs:
                x, x_ = conv([x, x_], edge_index)
                x, x_ = self.act(x), self.act(x_)
                x, x_ = F.dropout(x,  self.dropout, training=self.training), \
                        F.dropout(x_, self.dropout, training=self.training)
                hs.append(x)
                hs_.append(x_)

        h, alpha = self.summarize(hs, hs_) # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h), alpha


    def inference(self, x, loader, device, num_splits=10):
        # we do not use dropout because inferense is test
        root = utils.get_original_cwd()
        x_ = x.clone().detach()
        for i, conv in enumerate(self.convs):
            x, x_ = conv_for_gpumemory([x, x_], loader, conv, device)
            x, x_ = self.act(x), self.act(x_)
            np.save(root+'/data/tensor_buff/h_L{}.npy'.format(i), x.to('cpu').detach().numpy().copy())
            np.save(root+'/data/tensor_buff/h_twin_L{}.npy'.format(i), x_.to('cpu').detach().numpy().copy())

        num_nodes = x.size()[0]
        partition_size = int(num_nodes / num_splits)
        h_all, alpha_all = [], []
        for i in range(num_splits):
            start_idx, end_idx = partition_size*i, partition_size*(i+1)
            if i == num_splits-1:
                end_idx = num_nodes
            hs, hs_ = [], []
            for l in range(len(self.convs)):
                h  = np.load(root+'/data/tensor_buff/h_L{}.npy'.format(l))      # (n, d)
                h_ = np.load(root+'/data/tensor_buff/h_twin_L{}.npy'.format(l)) # (n, d)
                h  = torch.from_numpy(h[start_idx:end_idx].astype(np.float32)).clone().to(device)
                h_ = torch.from_numpy(h_[start_idx:end_idx].astype(np.float32)).clone().to(device)
                hs.append(h)   # (partition_size, d)
                hs_.append(h_) # (partition_size, d)
            h, alpha = self.summarize(hs, hs_)
            h_all.append(h)
            alpha_all.append(alpha)
        h = torch.cat(h_all, axis=0)
        alpha = torch.cat(alpha_all, axis=0)

        return self.out_lin(h), alpha
