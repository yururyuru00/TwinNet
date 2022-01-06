import torch
import torch.nn as nn
import math

from torch.nn import Linear, LSTM
from torch.nn import LayerNorm, BatchNorm1d
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, norm,
                 n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()
        self.task = task
        self.conv_name = conv_name

        if conv_name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels)

        elif conv_name == 'sage_conv':
            self.conv  = SAGEConv(in_channels, out_channels)

        elif conv_name == 'gat_conv':
            if iscat[0]: # if previous gatconv's cat is True
                in_channels = in_channels * n_heads[0]
            self.conv  = GATConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 heads=n_heads[1],
                                 concat=iscat[1],
                                 dropout=dropout_att)
            if iscat[1]: # if this gatconv's cat is True
                out_channels = out_channels * n_heads[1]

        
        if self.task == 'inductive':  # if 'inductive', we use linear
            self.linear = nn.Linear(in_channels, out_channels)
        
        if norm == 'LayerNorm':
            self.norm = LayerNorm(out_channels)
        elif norm == 'BatchNorm1d':
            self.norm = BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()


    def forward(self, x, edge_index):
        if self.task == 'transductive':
            x = self.conv(x, edge_index)
        elif self.task == 'inductive':
            x = self.conv(x, edge_index) + self.linear(x)

        return self.norm(x)



class TwinGNNConv(nn.Module):
    def __init__(self, model:GNNConv):
        super(TwinGNNConv, self).__init__()
        self.task = model.task

        if model.conv_name == 'gcn_conv':
            self.conv = model.conv.lin

        elif model.conv_name == 'sage_conv':
            raise NotImplementedError()

        elif model.conv_name == 'gat_conv':
            self.conv = model.conv.lin_src

        if self.task == 'inductive': # if inductive, we use linear
            self.linear = model.linear
        self.norm = model.norm

    def forward(self, x):
        if self.task == 'transductive':
            x = self.conv(x)
        elif self.task == 'inductive':
            x = self.conv(x) + self.linear(x)

        return self.norm(x)
        


class Summarize(nn.Module):
    def __init__(self, channels, scope, att_mode):
        super(Summarize, self).__init__()

        self.scope = scope
        self.att_mode = att_mode
        self.att = Linear(2 * channels, 1)
        self.att.reset_parameters()

    def forward(self, hs, hs_):
        h = torch.stack(hs, dim=1)  # h is (n, L, d).
        h_ = torch.stack(hs_, dim=1)  # h_ is also (n, L, d).
        
        if self.scope == 'local':
            query, key = h_, h
        else: # if 'global'
            query = h
            key = torch.mean(h_, dim=0, keepdim=True).repeat((2708, 1, 1))
        

        # 'Attention' takes query and key as input, alpha as output
        if self.att_mode == 'ad':
            query_key = torch.cat([query, key], dim=-1)
            alpha = self.att(query_key).squeeze(-1)

        elif self.att_mode == 'dp':
            alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])
            
        else: # if 'mx' (mix of ad and dp attention)
            query_key = torch.cat([query, key], dim=-1)
            alpha_ad = self.att(query_key).squeeze(-1)
            alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

        alpha_softmax = torch.softmax(alpha, dim=-1)
        return (h * alpha_softmax.unsqueeze(-1)).sum(dim=1) # h_i = \sum_l alpha_i^l * h_i^l



class SkipConnection(nn.Module):
    def __init__(self, skip_connection, n_hidden):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection
        if self.skip_connection == 'highway':
            self.linear = Linear(n_hidden, n_hidden)

    def forward(self, h, x):
        if self.skip_connection == 'vanilla':
            return h

        elif self.skip_connection == 'res':
            return h + x

        elif self.skip_connection == 'dense':
            return torch.cat([h, x], dim=-1)
            
        elif self.skip_connection == 'highway':
            gating_weights = torch.sigmoid(self.linear(x))
            ones = torch.ones_like(gating_weights)
            return h*gating_weights + x*(ones-gating_weights) # h*W + x*(1-W)
