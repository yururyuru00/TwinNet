from numpy import dtype, indices
from sympy import im
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from models.model_loader import load_net
from torch_geometric.nn import GCNConv  # noqa


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


adj_data   = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/adj_data.npy'))
adj_ind_s  = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/adj_indices.npy'))
adj_indptr = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/adj_indptr.npy'))
adj_shape  = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/adj_shape.npy'))

attr_data   = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/attr_data.npy'))
attr_ind_f  = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/attr_indices.npy'))
attr_indptr = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/attr_indptr.npy'))
attr_shape  = torch.Tensor(np.load('./data/syn-cora/h0.00-r2/attr_shape.npy'))

labels = torch.Tensor(np.load('./data/syn-cora/h0.20-r2/labels.npy')).to(torch.int64)

adj_ind_t = []
for i in range(adj_indptr.size()[0]-1):
    num_edge = int(adj_indptr[i+1] - adj_indptr[i])
    adj_ind_t.append(torch.full((1, num_edge), i, dtype=torch.int64))
adj_ind_t = torch.cat(adj_ind_t, dim=1).flatten()

attr_ind_v = []
for i in range(attr_indptr.size()[0]-1):
    num_feature = int(attr_indptr[i+1] - attr_indptr[i])
    attr_ind_v.append(torch.full((1, num_feature), i, dtype=torch.float32))
attr_ind_v = torch.cat(attr_ind_v, dim=1).flatten()

edge_index = torch.stack([adj_ind_s, adj_ind_t], dim=0).to(torch.int64)
attr_ind = torch.stack([attr_ind_v, attr_ind_f], dim=0)
n_nodes, n_features = int(attr_shape[0]), int(attr_shape[1])
x = torch.sparse_coo_tensor(attr_ind.tolist(), attr_data.tolist(), (n_nodes, n_features))
x = x.to_dense()
data = Data(x=x, edge_index=edge_index, y=labels)
print(data)

data.train_mask = torch.full((1, n_nodes), True).flatten()
data.valid_mask = torch.full((1, n_nodes), True).flatten()
data.test_mask = torch.full((1, n_nodes), True).flatten()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    acc = test()
    print(acc)
