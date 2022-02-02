import mlflow

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm

from models.model_loader import load_net


def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    out, _ = model(data.x_dict, data.adj_t_dict)
    out = out.log_softmax(dim=-1)
    y_true = data.y_dict['paper']
    loss = F.nll_loss(out[data.train_idx], y_true[data.train_idx].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, evaluator):
    model.eval()

    out, _ = model(data.x_dict, data.adj_t_dict)
    y_pred = out.argmax(dim=-1, keepdim=True)

    y_true = data.y_dict['paper']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_idx],
        'y_pred': y_pred[data.valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_idx],
        'y_pred': y_pred[data.test_idx],
    })['acc']

    return valid_acc, test_acc



def train_and_test(tri, cfg, data, device):
    model = load_net(cfg, num_nodes_dict = data.num_nodes_dict,
                          x_types = list(data.x_dict.keys()),
                          edge_types = list(data.adj_t_dict.keys()))
    model = model.to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator(name='ogbn-mag')

    for epoch in tqdm(range(1, cfg['epochs']+1)):
        train(model, data, optimizer)
        valid_acc, test_acc = test(model, data, evaluator)
        mlflow.log_metric(str(tri)+'th_valid_acces', value=valid_acc, step=epoch)
        mlflow.log_metric(str(tri)+'th_test_acces', value=test_acc, step=epoch)
    return test(model, data, evaluator)


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-mag', root+'/data/'+cfg.dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_year_dict = None
    data.edge_reltype_dict = None

    # Convert to new transposed `SparseTensor` format and add reverse edges.
    data.adj_t_dict = {}
    for keys, (row, col) in data.edge_index_dict.items():
        sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
        adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
        # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
        if keys[0] != keys[2]:
            data.adj_t_dict[keys] = adj.t()
            data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
        else:
            data.adj_t_dict[keys] = adj.to_symmetric()
    data.edge_index_dict = None

    data.train_idx = split_idx['train']['paper'].to(device)
    data.valid_idx = split_idx['valid']['paper'].to(device)
    data.test_idx = split_idx['test']['paper'].to(device)
    data = data.to(device)

    valid_acces, test_acces = [], []
    for tri in range(cfg['n_tri']):
        valid_acc, test_acc = train_and_test(tri, cfg, data, device)
        valid_acces.append(valid_acc)
        test_acces.append(test_acc)

    return valid_acces, test_acces, None