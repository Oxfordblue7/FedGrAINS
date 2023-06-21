import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, SAGEConv, GATConv
import random


"""
Server Models
"""
class serverSAGE(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverSAGE, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

class serverGAT(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGAT, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GATConv(nhid, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(GATConv(nhid, nhid))

class serverGCN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGCN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(nhid, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

"""
Client Models
"""
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, heads=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, heads, heads=heads)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(heads * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.nclass = out_channels

    def forward(self, inp):
        x = F.elu(self.conv1(inp.x, inp.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, inp.edge_index)
        return F.log_softmax(x, dim=-1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GCN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class SAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(SAGE, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
