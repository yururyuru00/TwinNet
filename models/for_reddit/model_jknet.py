import torch.nn as nn
import torch.nn.functional as F

from ..layer import GNNConv, conv_for_gpumemory
from torch_geometric.nn import JumpingKnowledge


# JKSAGE do not use skip-connection because SAGE already use skip-connection (h = AxW + xW_)
class JKSAGE(nn.Module):
    def __init__(self, cfg):
        super(JKSAGE, self).__init__()
        self.dropout = cfg.dropout
        self.act = eval(f'nn.' + cfg.activation + '()') # ReLU or Identity

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

    def forward(self, x, adjs, batch_size):
        hs, hs_ = [], []
        x_ = x.clone().detach()[:batch_size]
        for l, (edge_index, _, size) in enumerate(adjs): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, x_    = self.convs[l]([(x, x_target), x_], edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
            x, x_    = self.act(x), self.act(x_)
            x, x_    = F.dropout(x,  self.dropout, training=self.training), \
                       F.dropout(x_, self.dropout, training=self.training)
            hs.append(x)
            hs_.append(x_)
        hs  = [h[:batch_size] for h in hs]

        h = self.jk(hs, hs_) # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
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
