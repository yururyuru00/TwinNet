def load_net(cfg):
    if cfg.dataset == 'Reddit':
        from .for_reddit.model_ours import TwinSAGE
        from .for_reddit.model_jknet import JKSAGE
        from .for_reddit.model_gnn import SAGE

        # our algorithm
        if cfg.global_skip_connection == 'twin':
            return TwinSAGE(cfg)

        # existing algorithms
        elif cfg.global_skip_connection == 'jk':
            return JKSAGE(cfg)

        # existing algorithms
        else: # if global_skip_connection == 'vanilla'
            return SAGE(cfg)


    else: # if other datasets
        from .for_others.model_ours import TwinGCN, TwinGAT, TwinSAGE
        from .for_others.model_jknet import JKGCN, JKGAT, JKSAGE
        from .for_others.model_gnn import GCN, GAT, SAGE

        # our algorithm (attention skip-connection)
        if cfg.global_skip_connection == 'twin':
            if cfg.base_gnn == 'GCN':
                return TwinGCN(cfg)
            elif cfg.base_gnn == 'SAGE':
                return TwinSAGE(cfg)
            elif cfg.base_gnn == 'GAT':
                return TwinGAT(cfg)

        # existing algorithms
        elif cfg.global_skip_connection == 'jk':
            if cfg.base_gnn == 'GCN':
                return JKGCN(cfg)
            elif cfg.base_gnn == 'SAGE':
                return JKSAGE(cfg)
            elif cfg.base_gnn == 'GAT':
                return JKGAT(cfg)
    
        # existing algorithms
        else: # if global_skip_connection == 'vanilla'
            if cfg.base_gnn == 'GCN':
                return GCN(cfg)
            elif cfg.base_gnn == 'SAGE':
                return SAGE(cfg)
            elif cfg.base_gnn == 'GAT':
                return GAT(cfg)