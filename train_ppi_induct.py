from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from model import return_net


def train(loader, model, optimizer, device):
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criteria(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader, model, device):
    model.eval()

    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        ys.append(data.y)
        out = model(data.x, data.edge_index)
        preds.append((out > 0).float().cpu())

    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def train_and_test(cfg, data_loader, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(train_loader, model, optimizer, device)
    test_acc = test(test_loader, model, device)

    return test_acc


def run(cfg, root, device):
    train_dataset = PPI(root+'/data/'+cfg.dataset, split='train')
    val_dataset   = PPI(root+'/data/'+cfg.dataset, split='val')
    test_dataset  = PPI(root+'/data/'+cfg.dataset, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = train_and_test(cfg, data_loader, device)
        test_acces.append(test_acc)

    return test_acces