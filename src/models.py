import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Union

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

class GCN(torch.nn.Module):
    def __init__(self, nlayer, nfeat, nhid, ncls, dropping_method: str = "DropEdge", drop_rate: float = 0, args : dict = None):
        super(GCN, self).__init__()
        self.dropping_method = dropping_method
        self.drop_rate = drop_rate
        self.edge_weight = None
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

    def forward(self, x: Tensor, edge_index: Adj):
        for l in range(len(self.graph_convs)):
            x = self.graph_convs[l](x, edge_index)
            x = F.relu(x)
            if self.dropping_method == "Dropout":
                x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

class MaskedGCN(torch.nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args
        
        from layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x

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
                proxy: bool = False) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = torch.relu(layer(x, edges))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if proxy:
            return x

        edges = edge_index[0] if layerwise_adjacency else edge_index
        logits = self.gcn_layers[-1](x, edges)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        # torch.cuda.synchronize()
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc


class MaskedGCNv2(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0. , l1: float=1e-3, args : dict = None):
        super(MaskedGCNv2, self).__init__()
        from .layers import MaskedGCNConv
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(MaskedGCNConv(in_channels=dims[i],
                                            out_channels=dims[i + 1],cached=False, l1=l1, args = args) )
        #TODO: Check Later
        gcn_layers.append(MaskedGCNConv(in_channels=dims[-2], out_channels=dims[-1], l1=l1, args =args))
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                proxy: bool = False) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = torch.relu(layer(x, edges))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if proxy:
            return x

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
        self.drop_rate = drop_rate
        self.nlayer = nlayer
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nfeat, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))
        self.classifier = torch.nn.Linear(nhid, ncls)

    def forward(self, x: Tensor, edge_index: Adj):
        for l in range(len(self.graph_convs)):
            x = self.graph_convs[l](x, edge_index)
            x = F.relu(x)
            if self.dropping_method == "Dropout":
                x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
    
