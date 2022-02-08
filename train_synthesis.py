from tqdm import tqdm
import numpy as np
import mlflow

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MixHopSyntheticDataset

from models.model_loader import load_net
from utils import fix_seed, accuracy


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()

    h, _ = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train  = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h, _ = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val, _ = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])

    return loss_val.item(), acc_val


def test(data, model):
    model.eval()
    h, alpha = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc, _ = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    _, whole_node_correct = accuracy(prob_labels_test, data.y)

    return acc, alpha, whole_node_correct


def train_and_test(cfg, data, device):
    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg.learning_rate, 
                                 weight_decay = cfg.weight_decay)

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, cfg.epochs+1):
        loss_val, acc_val = train(data, model, optimizer)

        if loss_val < best_loss:
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == cfg.patience:
            break
    acc_test, alpha, whole_node_correct = test(data, model)

    return acc_val, acc_test, alpha, whole_node_correct


def run(cfg, root, device):
    valid_acces, test_acces, artifacts = [], [], {}
    for tri in tqdm(range(cfg.n_tri)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = MixHopSyntheticDataset(root      = root+'/data/syn/h{}'.format(cfg.homophily),
                                         homophily = cfg.homophily)
        data = dataset[0].to(device)

        valid_acc, test_acc, alpha, correct = train_and_test(cfg, data, device)
        valid_acces.append(valid_acc.to('cpu').item())
        test_acces.append(test_acc.to('cpu').item())
        artifacts['alpha_{}.npy'.format(tri)] = alpha
        artifacts['correct_{}.npy'.format(tri)] = correct
        artifacts['test_mask_{}.npy'.format(tri)] = data.test_mask

    return valid_acces, test_acces, artifacts