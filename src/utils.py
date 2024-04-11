import os
import argparse
import pickle
import random
import time
import torch
import numpy as np
import networkx as nx
from torch import Tensor
import scipy.sparse as sp
from typing import Tuple, Dict
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch_geometric.datasets as datasets
from torch.distributions import Bernoulli, Gumbel
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_networkx, degree, subgraph, to_scipy_sparse_matrix

#from community import community_louvain
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import BaseTransform


def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def get_data(dataset, data_path):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    elif dataset in ['Computers', 'Photo']:
        data = datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    return data

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graph , train=None, test=None, shuffle=True, seed=None):
    y = graph.y
    graphs_tv, graphs_test = train_test_split(torch.arange(0, len(y)), train_size=train, test_size=test, shuffle=shuffle, random_state=seed)
    subgraph_tr = subgraph(torch.LongTensor(graphs_tv), graph.edge_index)[0]
    subgraph_test = subgraph(torch.LongTensor(graphs_test), graph.edge_index)[0]
    return graph.subgraph(subgraph_tr), graph.subgraph(subgraph_test)


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    # avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    # df.loc[ds, 'avgNodes_train'] = avgNodes
    # df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        # avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        # df.loc[ds, 'avgNodes_val'] = avgNodes
        # df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        # avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        # df.loc[ds, 'avgNodes_test'] = avgNodes
        # df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove edges from an edge index, by removing nodes according to some
    probability.

    Uses Gumbel-max trick to sample from Bernoulli distribution. This is off-policy, since the original input
    distribution is a regular Bernoulli distribution.
    Args:
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """

    k = num_samples
    n = neighbor_nodes.shape[0]
    if k >= n:
        # TODO: Test this setting
        # print("here")
        return neighbor_nodes, torch.sigmoid(
            logits.squeeze(-1) + 1e-7).log(), {}
    assert k < n
    assert k > 0
    # print(logits)
    b = Bernoulli(logits=logits.squeeze())

    # Gumbel-sort trick https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    gumbel = Gumbel(torch.tensor(0., device=logits.device), torch.tensor(1., device=logits.device))
    gumbel_noise = gumbel.sample((n,))
    perturbed_log_probs = b.probs.log() + gumbel_noise

    samples = torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1]

    p = b.probs.clone().detach()
    # calculate the entropy in bits
    entropy = -(p * (p).log2() + (1 - p) * (1 - p).log2())

    min_prob = p.min(-1)[0]
    max_prob = p.max(-1)[0]

    if torch.isnan(entropy).any():
        nan_ind = torch.isnan(entropy)
        entropy[nan_ind] = 0.0

    std_entropy, mean_entropy = torch.std_mean(entropy)
    mask = torch.zeros_like(logits.squeeze(), dtype=torch.float)
    mask[samples] = 1

    neighbor_nodes = neighbor_nodes[mask.bool().cpu()]

    stats_dict = {"min_prob": min_prob,
                  "max_prob": max_prob,
                  "mean_entropy": mean_entropy,
                  "std_entropy": std_entropy}

    return neighbor_nodes, b.log_prob(mask), stats_dict

def get_neighborhoods(nodes: Tensor,
                      adjacency: sp.csr_matrix
                      ) -> Tensor:
    """Returns the neighbors of a set of nodes from a given adjacency matrix"""
    neighborhood = adjacency[nodes].tocoo()
    neighborhoods = torch.stack([nodes[neighborhood.row],
                                 torch.tensor(neighborhood.col)],
                                dim=0)
    return neighborhoods

def slice_adjacency(adjacency: sp.csr_matrix, rows: Tensor, cols: Tensor):
    """Selects a block from a sparse adjacency matrix, given the row and column
    indices. The result is returned as an edge index.
    """
    row_slice = adjacency[rows]
    row_col_slice = row_slice[:, cols]
    slice = row_col_slice.tocoo()
    edge_index = torch.stack([rows[slice.row],
                              cols[slice.col]],
                             dim=0)
    return edge_index

def get_sparsity(masks, l1):
    n_active, n_total = 0, 1
    for mask in masks:
        pruned = torch.abs(mask) < l1
        mask = torch.ones(mask.shape).cuda().masked_fill(pruned, 0)
        n_active += torch.sum(mask)
        _n_total = 1
        for s in mask.shape:
            _n_total *= s 
        n_total += _n_total
    return ((n_total-n_active)/n_total).item()


class TensorMap:
    """A class used to quickly map integers in a tensor to an interval of
    integers from 0 to len(tensor) - 1. This is useful for global to local
    conversions.

    Example:
        >>> nodes = torch.tensor([22, 32, 42, 52])
        >>> node_map = TensorMap(size=nodes.max() + 1)
        >>> node_map.update(nodes)
        >>> node_map.map(torch.tensor([52, 42, 32, 22, 22]))
        tensor([3, 2, 1, 0, 0])
    """

    def __init__(self, size):
        self.map_tensor = torch.empty(size, dtype=torch.long)
        self.values = torch.arange(size)

    def update(self, keys: Tensor):
        values = self.values[:len(keys)]
        self.map_tensor[keys] = values

    def map(self, keys):
        return self.map_tensor[keys]


class LargestConnectedComponents(BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph.

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
    """
    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'