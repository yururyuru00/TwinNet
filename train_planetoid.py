from tqdm import tqdm
import numpy as np
import mlflow

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.model_loader import load_net


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels), correct


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

    return loss_val.item()


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
    for epoch in range(1, cfg.epochs):
        loss_val = train(data, model, optimizer)

        if loss_val < best_loss:
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == cfg.patience:
            break

    return test(data, model)


def run(cfg, root, device):
    dataset = Planetoid(root          = root + '/data/' + cfg.dataset,
                        name          = cfg.dataset,
                        split         = cfg.split,
                        transform     = eval(cfg.transform))
    data = dataset[0].to(device)

    test_acces, artifacts = [], {}
    for tri in tqdm(range(cfg.n_tri)):
        test_acc, alpha, correct = train_and_test(cfg, data, device)
        test_acces.append(test_acc.to('cpu').item())
        artifacts['alpha_{}.npy'.format(tri)] = alpha
        artifacts['correct_{}.npy'.format(tri)] = correct

    return test_acces, artifacts