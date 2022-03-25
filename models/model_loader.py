from .twingnn import TwinGCN, TwinGAT, TwinSAGE
from .jknet import JKGCN, JKGAT, JKSAGE
from .gnn import GCN, GAT, SAGE
    

def load_net(cfg, **kwargs):

    # our algorithm (TwinGNN)
    if cfg.global_skip_connection == 'twin':
        if cfg.base_gnn == 'GCN':
            return TwinGCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return TwinSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return TwinGAT(cfg)
    
    # existing algorithms (JKNet)
    elif cfg.global_skip_connection == 'jk':
        if cfg.base_gnn == 'GCN':
            return JKGCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return JKSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return JKGAT(cfg)
    
    # existing algorithms (GNN)
    else:
        if cfg.base_gnn == 'GCN':
            return GCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return SAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return GAT(cfg)