import torch.nn as nn
import torch.nn.functional as F

from ..layer import GNNConv, conv_for_gpumemory


# SAGE do not use skip-connection because SAGE already use skip-connection (h = AxW + xW_)
class SAGE(nn.Module):
    
    def __init__(self, cfg):
        super(SAGE, self).__init__()
        self.dropout = cfg.dropout

        self.convs = nn.ModuleList()
        self.convs(GNNConv('sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm))
        for l in range(1, cfg.n_layer-1):
            self.convs.append(GNNConv('sage_conv', cfg.n_hid, cfg.n_hid, cfg.norm))
        self.out_conv = GNNConv('sage_conv', cfg.n_hid, cfg.n_class, norm='None')


    def forward(self, x, adjs, _):
        if isinstance(adjs, list):
            adjs, out_adj = adjs[:-1], adjs[-1]

            for adj, conv in zip(adjs, self.convs): # size is [B_l's size, B_(l+1)'s size]
                x_target = x[:adj.size[1]]
                x = conv((x, x_target), adj.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_target = x[:out_adj.size[1]]
            x = self.out_conv((x, x_target), out_adj.edge_index)

        else:
            edge_index = adjs

            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)

        return x, None


    def inference(self, x, loader, device):
        # we do not use dropout because inferense is test
        x = conv_for_gpumemory(x, loader, self.in_conv, device)
        x = F.relu(x)

        for mid_conv in self.mid_convs:
            x = conv_for_gpumemory(x, loader, mid_conv, device)
            x = F.relu(x)

        x = conv_for_gpumemory(x, loader, self.out_conv, device)
        
        return x, None
