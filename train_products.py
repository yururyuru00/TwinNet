import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models.model_loader import load_net


def train(loader, model, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, alpha = model(data.x, data.edge_index)
        pred = torch.log_softmax(out, dim=-1)
        y = data.y.squeeze(1)
        loss = F.nll_loss(pred[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()



@torch.no_grad()
def test(data, loader, model, evaluator, device):
    model.eval()

    out, alpha = model.inference(data.x, loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return valid_acc, test_acc


def train_and_test(tri, cfg, data, device):
    # data initialize each tri
    torch.manual_seed(cfg.seed + tri)
    torch.cuda.manual_seed(cfg.seed + tri)
    train_loader = GraphSAINTRandomWalkSampler(data,
                                               batch_size=20000,
                                               walk_length=cfg.n_layer,
                                               num_steps=30,
                                               sample_coverage=0,
                                               save_dir=data.processed_dir)
    test_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=0)

    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator(name='ogbn-products')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(train_loader, model, optimizer, device)

    return test(data, test_loader, model, evaluator, device)


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-products', root+'/data/'+cfg.dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.processed_dir = dataset.processed_dir

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    valid_acces, test_acces = [], []
    for tri in range(cfg['n_tri']):
        valid_acc, test_acc = train_and_test(tri, cfg, data, device)
        valid_acces.append(valid_acc)
        test_acces.append(test_acc)

    return valid_acces, test_acces, None