import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn import Linear, LSTM
from torch.nn import LayerNorm, BatchNorm1d, Identity
from torch_geometric.nn import GATConv, GCNConv, SAGEConv



# gnn convolution for gpumemory, this is used for only reddit dataset
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



class GNNConv(nn.Module):
    def __init__(self, conv_name, in_channels, out_channels, norm,
                 self_loop=True, n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()

        if conv_name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels, add_self_loops=self_loop)
            self.conv_ = self.conv.lin

        elif conv_name == 'sage_conv':
            self.conv  = SAGEConv(in_channels, out_channels, root_weight=self_loop)
            self.lin_root     = self.conv.lin_r
            self.lin_neighbor = self.conv.lin_l
            self.conv_ = lambda x_: self.lin_root(x_) + self.lin_neighbor(x_)

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

        alpha = torch.softmax(alpha/self.att_temparature, dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1), alpha



class JumpingKnowledge(torch.nn.Module):
    def __init__(self, mode, channels=None, num_layers=None):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(channels, (num_layers * channels) // 2,
                             bidirectional=True, batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1), None
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0], None
        elif self.mode == 'lstm':
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1), alpha



class SkipConnection(nn.Module):
    def __init__(self, skip_connection, in_channels, out_channels):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection

        if in_channels == out_channels:
            self.transformer = Identity()
        else:
            self.transformer = Linear(in_channels, out_channels)

        if self.skip_connection == 'highway':
            self.gate_linear = Linear(out_channels, out_channels)

    def forward(self, h_x):
        if isinstance(h_x, list): # if h_x_ is [(h,x), (h_,x_)], we use twin-skip
            h,  x  = h_x[0]
            h_, x_ = h_x[1]
    
            if self.skip_connection == 'vanilla':
                return h, h_

            else: # if use any skip_connection
                x  = self.transformer(x) # in_channels >> out_channels
                x_ = self.transformer(x_)

                if self.skip_connection == 'res':
                    return h + x, h_ + x_

                elif self.skip_connection == 'dense':
                    return torch.cat([h, x], dim=-1), torch.cat([h_, x_], dim=-1)

                elif self.skip_connection == 'highway':
                    gating_weights = torch.sigmoid(self.gate_linear(x))
                    ones = torch.ones_like(gating_weights)
                    return h*gating_weights + x*(ones-gating_weights), \
                           h_*gating_weights + x_*(ones-gating_weights)

        else: # if h_x_ is (h,x), we use single-skip
            h,  x  = h_x

            if self.skip_connection == 'vanilla':
                return h

            else: # if use any skip_connection
                x  = self.transformer(x) # in_channels >> out_channels

                if self.skip_connection == 'res':
                    return h + x

                elif self.skip_connection == 'dense':
                    return torch.cat([h, x], dim=-1)

                elif self.skip_connection == 'highway':
                    gating_weights = torch.sigmoid(self.gate_linear(x))
                    ones = torch.ones_like(gating_weights)
                    return h*gating_weights + x*(ones-gating_weights)



def orthonomal_loss(model, device):
    def eyes_like(tensor):
        size = tensor.size()[0]
        return torch.eye(size, out=torch.empty_like(tensor)).to(device)

    orthonomal_loss = torch.tensor(0, dtype=torch.float32).to(device)
    for conv in model.convs: # in [W^1, W^2, ... , W^L]
        weight = conv.conv.lin_src.weight
        mm = torch.mm(weight, torch.transpose(weight, 0, 1))
        orthonomal_loss += torch.norm(mm - eyes_like(mm))

    return orthonomal_loss