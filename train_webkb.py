from termios import ECHOPRT
from tqdm import tqdm
import mlflow
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import WebKB

from models.model_loader import load_net
from utils import accuracy


def train(tri, data, model, optimizer):
    model.train()
    optimizer.zero_grad()

    h, alpha = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train  = F.nll_loss(prob_labels[data.train_mask[:,tri]], data.y[data.train_mask[:,tri]])
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h, alpha = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask[:,tri]], data.y[data.val_mask[:,tri]])
    acc_val, _ = accuracy(prob_labels_val[data.val_mask[:,tri]], data.y[data.val_mask[:,tri]])

    return loss_val.item(), acc_val


def test(tri, data, model):
    model.eval()
    h, alpha = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc, _ = accuracy(prob_labels_test[data.test_mask[:,tri]], data.y[data.test_mask[:,tri]])

    return acc


def train_and_test(tri, cfg, data, device):
    model = load_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg.learning_rate, 
                                 weight_decay = cfg.weight_decay)

    loss_val_min = np.inf
    acc_val_max = 0.0
    for epoch in range(1, cfg.epochs+1):
        loss_val, acc_val = train(tri, data, model, optimizer)
        acc_test = test(tri, data, model)
        # mlflow.log_metric(str(tri)+'th_valid_acces', value=acc_val.item(), step=epoch)
        # mlflow.log_metric(str(tri)+'th_test_acces', value=acc_test.item(), step=epoch)

        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        if acc_val >= acc_val_max or loss_val <= loss_val_min:
            acc_val_max = np.max((acc_val, acc_val_max))
            loss_val_min = np.min((loss_val, loss_val_min))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= cfg.patience:
                break

    acc_test = test(tri, data, model)
    return acc_val, acc_test


def run(cfg, root, device):
    dataset = WebKB(root      = root + '/data/' + cfg.dataset,
                    name      = cfg.dataset)
    data = dataset.data.to(device)

    valid_acces, test_acces = [], []
    for tri in tqdm(range(cfg.n_tri)):
        valid_acc, test_acc = train_and_test(tri, cfg, data, device)
        valid_acces.append(valid_acc.to('cpu').item())
        test_acces.append(test_acc.to('cpu').item())

    return valid_acces, test_acces, None