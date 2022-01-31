def load_net(cfg):
    if cfg.dataset in ['Reddit', 'Products']:
        from .for_large_scale.model_ours import TwinSAGE
        from .for_large_scale.model_jknet import JKSAGE
        from .for_large_scale.model_gnn import SAGE

        # our algorithm
        if cfg.global_skip_connection == 'twin':
            return TwinSAGE(cfg)

        # existing algorithms
        elif cfg.global_skip_connection == 'jk':
            return JKSAGE(cfg)

        # existing algorithms
        else: # if global_skip_connection == 'vanilla'
            return SAGE(cfg)


    else: # if other small-scale datasets
        from .for_small_scale.model_ours import TwinGCN, TwinGAT, TwinSAGE
        from .for_small_scale.model_jknet import JKGCN, JKGAT, JKSAGE
        from .for_small_scale.model_gnn import GCN, GAT, SAGE

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