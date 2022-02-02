def load_net(cfg, **kwargs):

    if cfg.data_type == 'heterogeneous':
        from .for_hetero_data.model_ours import TwinRGCN
        from .for_hetero_data.model_jknet import JKRGCN
        from .for_hetero_data.model_gnn import RGCN

        num_nodes_dict = kwargs.get('num_nodes_dict')
        x_types        = kwargs.get('x_types')
        edge_types      = kwargs.get('edge_types')
        # our algorithm
        if cfg.global_skip_connection == 'twin':
            return TwinRGCN(cfg, num_nodes_dict, x_types, edge_types)

        # existing algorithms (jk-net)
        elif cfg.global_skip_connection == 'jk':
            return JKRGCN(cfg, num_nodes_dict, x_types, edge_types)

        # existing algorithms (gnn)
        else:
            return RGCN(cfg, num_nodes_dict, x_types, edge_types)


    elif cfg.data_type == 'large-scale':
        from .for_large_scale.model_ours import TwinSAGE
        from .for_large_scale.model_jknet import JKSAGE
        from .for_large_scale.model_gnn import SAGE

        # our algorithm
        if cfg.global_skip_connection == 'twin':
            return TwinSAGE(cfg)

        # existing algorithms (jk-net)
        elif cfg.global_skip_connection == 'jk':
            return JKSAGE(cfg)

        # existing algorithms (gnn)
        else:
            return SAGE(cfg)


    else: # if data_type == 'small-scale'
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