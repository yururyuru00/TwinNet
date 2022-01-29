from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models.model_loader import load_net


def acc(y_true, y_pred):
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:,i] == y_true[:,i]
        correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
    return correct.to(torch.int8)


def train(data, model, optimizer):
    model.train()

    optimizer.zero_grad()
    out, alpha = model(data.x, data.adj_t)
    out = out.log_softmax(dim=-1)
    out = out[data['train_mask']]
    loss = F.nll_loss(out, data.y.squeeze(1)[data['train_mask']])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data, model, evaluator):
    model.eval()
    out, alpha = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    mask = data['test_mask']
    test_acc = evaluator.eval({'y_true': data.y[mask],
                               'y_pred': y_pred[mask],})['acc']
    whole_correct = acc(data.y, y_pred)
    
    return test_acc, alpha, whole_correct


def train_and_test(cfg, data, device):
    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-arxiv')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(data, model, optimizer)
    test_acc = test(data, model, evaluator)

    return test_acc


def run(cfg, root, device):
    dataset = PygNodePropPredDataset('ogbn-arxiv', root + '/data/' + cfg.dataset, 
                                     transform=T.ToSparseTensor())
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    test_acces, artifacts = [], {}
    for tri in range(cfg['n_tri']):
        test_acc, alpha, correct = train_and_test(cfg, data, device)
        test_acces.append(test_acc)
        artifacts['alpha_{}.npy'.format(tri)] = alpha
        artifacts['correct_{}.npy'.format(tri)] = correct

    return test_acces, artifacts