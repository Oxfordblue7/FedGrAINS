import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Union
"""
Unified DropBlock Module for Dropout, DropEdge, and DropNode
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
    def __init__(self, nlayer, nfeat, nhid, ncls):
        super(serverSAGE, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

# class serverGAT(torch.nn.Module):
#     def __init__(self, nlayer, nfeat, nhid, ncls):
#         super(serverGAT, self).__init__()
#         self.graph_convs = torch.nn.ModuleList()
#         self.graph_convs.append(GATConv(nfeat, nhid))
#         for l in range(nlayer - 1):
#             self.graph_convs.append(GATConv(nhid, nhid))
#         self.classifier = torch.nn.Linear(nhid, ncls)

class serverGCN(torch.nn.Module):
    def __init__(self, nlayer, nfeat, nhid, ncls):
        super(serverGCN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

class serverGCNv2(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int]):
        super(GCNv2, self).__init__()
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)


"""
Client Models
"""
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, dropping_method: str = "DropEdge", heads=2):
#         super(GAT, self).__init__()
#         self.dropping_method = dropping_method
#         self.drop_block = DropBlock(dropping_method)
#
#         self.conv1 = GATConv(in_channels, heads, heads=heads)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(heads * heads, out_channels, heads=1, concat=False, dropout=0.6)
#         self.nclass = out_channels
#
#     def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
#
#         if self.training:
#             x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)
#
#         x = F.elu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=-1)
#
#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)

class GCN(torch.nn.Module):
    def __init__(self, nlayer, nfeat, nhid, ncls, dropping_method: str = "DropEdge", drop_rate: float = 0):
        super(GCN, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        self.drop_rate = drop_rate
        self.edge_weight = None
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

    def forward(self, x: Tensor, edge_index: Adj):
        if self.training:
            if self.dropping_method != "Dropout":
                x, edge_index = self.drop_block.drop(x, edge_index, self.drop_rate)
        for l in range(len(self.graph_convs)):
            x = self.graph_convs[l](x, edge_index)
            x = F.relu(x)
            if self.dropping_method == "Dropout":
                x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

class GCNv2(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0.):
        super(GCNv2, self).__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = torch.relu(layer(x, edges))
            x = F.dropout(x, p=self.dropout, training=self.training)

        edges = edge_index[0] if layerwise_adjacency else edge_index
        logits = self.gcn_layers[-1](x, edges)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        # torch.cuda.synchronize()
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc


class SAGE(torch.nn.Module):
    def __init__(self, nlayer, nfeat, nhid, ncls, dropping_method: str = "DropEdge", drop_rate: float = 0):
        super(SAGE, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)
        self.drop_rate = drop_rate
        self.nlayer = nlayer
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

    def forward(self, x: Tensor, edge_index: Adj):
        if self.training:
            if self.dropping_method != "Dropout":
                x, edge_index = self.drop_block.drop(x, edge_index, self.drop_rate)
        for l in range(len(self.graph_convs)):
            x = self.graph_convs[l](x, edge_index)
            x = F.relu(x)
            if self.dropping_method == "Dropout":
                x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
    
