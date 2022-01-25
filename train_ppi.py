from tqdm import tqdm

import torch
from torch_scatter import scatter
from torch_geometric.loader import RandomNodeSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models.model_loader import load_net


def train(loader, model, optimizer, device):
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out, alpha = model(data.x, data.edge_index)
        loss = criteria(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader, model, evaluator, device):
    model.eval()

    ys, preds, alphas = [], [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        out, alpha = model(data.x, data.edge_index)
        mask = data['test_mask']
        ys.append(data.y[mask].cpu())
        preds.append(out[mask].cpu())

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(ys, dim=0),
        'y_pred': torch.cat(preds, dim=0),
    })['rocauc']

    return test_rocauc.item()


def train_and_test(tri, cfg, data, device):
    # data initialize each tri
    torch.manual_seed(cfg.seed + tri)
    torch.cuda.manual_seed(cfg.seed + tri)
    train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True,
                                     num_workers=0)
    test_loader = RandomNodeSampler(data, num_parts=5, shuffle=False, 
                                    num_workers=0)

    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-proteins')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(train_loader, model, optimizer, device)
    test_acc = test(test_loader, model, evaluator, device)

    return test_acc


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-proteins', root+'/data/'+cfg.dataset)
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    
    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
    cfg.n_feat = cfg.e_feat

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = train_and_test(tri, cfg, data, device)
        test_acces.append(test_acc)

    return test_acces, None