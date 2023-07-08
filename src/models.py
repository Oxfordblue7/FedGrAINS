import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.typing import Adj
from torch_geometric.nn import GCNConv, global_add_pool, SAGEConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import random


"""
Usified DropBlock Module for Dropout, DropEdge, and DropNode
"""
class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == 'DropNode':
            x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)
            x = x / (1 - drop_rate)
        elif self.dropping_method == 'DropEdge':
            edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
            if isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
                edge_index = torch.stack((row, col))
            perm = torch.randperm(edge_index.size(1))
            edge_index = edge_index.t()[perm][:edge_reserved_size].t()
        elif self.dropping_method == 'Dropout':
            x = F.dropout(x, drop_rate)
        else:
            #The case where we do not employ dropout methods at all 
            return x, edge_index

        return x, edge_index


"""
Server Models
"""
class serverSAGE(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverSAGE, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

class serverGAT(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGAT, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        for l in range(nlayer - 1):
            self.graph_convs.append(GATConv(nhid, nhid))

class serverGCN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGCN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

"""
Client Models
"""
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropping_method: str = "DropEdge", heads=2):
        super(GAT, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        self.conv1 = GATConv(in_channels, heads, heads=heads)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(heads * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.nclass = out_channels

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):

        if self.training:
            x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)

        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropping_method: str = "DropEdge"):
        super(GCN, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        self.edge_weight = None

        self.num_layers = nlayer
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.training:
            x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class SAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropping_method: str = "DropEdge"):
        super(SAGE, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        self.num_layers = nlayer
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.training:
            x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
