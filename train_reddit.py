from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

from model_reddit import return_net


def train(data, train_loader, model, optimizer, device):
    model.train()

    for batch_id, (batch_size, n_id, adjs) in enumerate(train_loader):
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        h = model(data.x[n_id], adjs, batch_size)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(data, test_loader, model, device):
    model.eval()

    h = model.inference(data.x, test_loader, device)
    y_true = data.y.unsqueeze(-1)
    y_pred = h.argmax(dim=-1, keepdim=True)
    test_acc = int(y_pred[data.test_mask].eq(y_true[data.test_mask]).sum()) / int(data.test_mask.sum())
    
    return test_acc


def train_and_test(tri, cfg, data, device):
    # data initialize each tri
    torch.manual_seed(cfg.seed + tri)
    torch.cuda.manual_seed(cfg.seed + tri)
    sizes_each_layer = [25, 10, 10, 10, 10, 10] # sampling size of each layer when aggregates
    train_loader = NeighborSampler(data.edge_index, 
                                   node_idx    = data.train_mask,
                                   sizes       = sizes_each_layer[:cfg['n_layer']], 
                                   batch_size  = 1024,
                                   shuffle     = True,
                                   num_workers = 0)
    test_loader = NeighborSampler(data.edge_index, 
                                  node_idx    = None,
                                  sizes       = [-1], 
                                  batch_size  = 1024, 
                                  shuffle     = False,
                                  num_workers = 0)

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    
    for epoch in tqdm(range(1, cfg['epochs'])):
        train(data, train_loader, model, optimizer, device)
    test_acc = test(data, test_loader, model, device)

    return test_acc


def run(cfg, root, device):
    dataset = Reddit(root = root+'/data/'+cfg.dataset)
    data = dataset[0].to(device)

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = train_and_test(tri, cfg, data, device)
        test_acces.append(test_acc)

    return test_acces
        