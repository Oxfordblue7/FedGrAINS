import os
import copy
import torch
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn import preprocessing
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix

from ogb.nodeproppred import PygNodePropPredDataset 

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

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

def split_train(data, dataset, data_path, ratio_train, mode, n_clients):
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True

    torch_save(f'{data_path}/{dataset}_{mode}/{n_clients}/','train.pt', {'data': data})
    torch_save(f'{data_path}/{dataset}_{mode}/{n_clients}/','test.pt', {'data': data})
    torch_save(f'{data_path}/{dataset}_{mode}/{n_clients}/','val.pt', {'data': data})
    print(f'Split done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data


def louvain_graph_cut(whole_graph, node_subjects, num_owners, delta = 20):
    edges = whole_graph.edges
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    G = nx.Graph()
    G.add_edges_from(edges)

    partition = community_louvain.best_partition(G)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])

    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = len(list(whole_graph.nodes())) // num_owners - delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}

    owner_node_ids = {owner_id: [] for owner_id in range(num_owners)}

    owner_nodes_len = len(list(G.nodes())) // num_owners
    owner_list = [i for i in range(num_owners)]
    owner_ind = 0

    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for ' + str(owner_i) + ' = ' + str(len(owner_node_ids[owner_i])))

    #Count number of lost edges due to partitioning
    num_edges = sum([len(list(whole_graph.subgraph(owner_node_ids[owner_i]).edges)) for owner_i in range(num_owners)])

    return len(list(edges)) - num_edges, owner_node_ids


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
