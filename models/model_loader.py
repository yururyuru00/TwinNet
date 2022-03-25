from .for_small_scale.model_ours import TwinGCN, TwinGAT, TwinSAGE
from .for_small_scale.model_jknet import JKGCN, JKGAT, JKSAGE
from .for_small_scale.model_gnn import GCN, GAT, SAGE
    

def load_net(cfg, **kwargs):

    # our algorithm (attention skip-connection)
    if cfg.global_skip_connection == 'twin':
        if cfg.base_gnn == 'GCN':
            return TwinGCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return TwinSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return TwinGAT(cfg)
    
    # existing algorithms (jk-net)
    elif cfg.global_skip_connection == 'jk':
        if cfg.base_gnn == 'GCN':
            return JKGCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return JKSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return JKGAT(cfg)
    
    # existing algorithms (gnn)
    else:
        if cfg.base_gnn == 'GCN':
            return GCN(cfg)
        elif cfg.base_gnn == 'SAGE':
            return SAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            return GAT(cfg)