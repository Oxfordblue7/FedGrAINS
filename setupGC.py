import random
from random import choices
import numpy as np
import pandas as pd
import sys
import torch
import metispy as metis
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.loader import DataLoader
from sklearn import preprocessing
from torch_geometric.utils import to_networkx, to_dense_adj, dense_to_sparse


from community import community_louvain

from src.models import *
from src.server import Server
from src.client import Client_GC
from src.utils import LargestConnectedComponents

def _metis_graph_cut(g, num_owners):
    G = to_networkx(g)
    n_cuts, membership = metis.part_graph(G, num_owners)
    assert len(list(set(membership))) == num_owners
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')
    return n_cuts, membership    

def _split_train(data, train_ratio = 0.2):
    n_data = data.num_nodes
    ratio_test = (1-train_ratio)/2
    n_train = round(n_data * train_ratio)
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
    return data

def prepareData_oneDS(datapath, data, num_client, batchSize, seed=None, overlap=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if data in ["Cora", "Citeseer", "Pubmed"]:
        pyg_dataset = Planetoid(f"{datapath}/Planetoid", data, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))
        num_classes= pyg_dataset.num_classes
        num_features  = pyg_dataset.num_features
        dataset = pyg_dataset[0]
    elif data in ['Computers', 'Photo']:
        pyg_dataset = Amazon(f"{datapath}/{data}", data, transform=T.Compose([T.NormalizeFeatures()]))
        num_classes= pyg_dataset.num_classes
        num_features  = pyg_dataset.num_features
        dataset = pyg_dataset[0]
        dataset.train_mask, dataset.val_mask, dataset.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    else:
        #MS_academic
        raise Exception("MS Academic not yet implemented!")

    #Train-val-test split
    #Train ratio is 0.2 in accordance with FedPUB
    dataset = _split_train(dataset)

    #Disjoint metis partitioning 
    _ , membership =  _metis_graph_cut(dataset, num_client)
    
    splitedData = {}
    df = pd.DataFrame()
    adj = to_dense_adj(dataset.edge_index)[0]
    for client_id in range(num_client):
        ds = f'{client_id}-{data}'

        #Distribute partition to the user
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        client_adj = adj[client_indices][:, client_indices]
        client_edge_index, _ = dense_to_sparse(client_adj)
        client_edge_index = client_edge_index.T.tolist()
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = dataset.x[client_indices]
        client_y = dataset.y[client_indices]
        client_train_mask = dataset.train_mask[client_indices]
        client_val_mask = dataset.val_mask[client_indices]
        client_test_mask = dataset.test_mask[client_indices]
        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index.t().contiguous(),
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        assert torch.sum(client_train_mask).item() > 0
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

        #Generate dataloaders
        #Couldnt run Cora with mini-batching
        dataloader =  DataLoader([client_data], batch_size=batchSize, shuffle=False)
        splitedData[ds] = (dataloader,num_features, num_classes, client_num_nodes)
        #df = get_stats(df, ds, train_data, graphs_val=val_data, graphs_test=test_data)

    return splitedData, num_classes

def setup_devices(splitedData, num_classes, args):
    idx_clients = {}
    clients = []
    num_node_features = None
    num_graph_labels = None
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloader, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = getattr(sys.modules[__name__], args.model)(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropping_method)
        optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloader, optimizer, args))

    #Create server model
    smodel = getattr(sys.modules[__name__], "server"+args.model)(nlayer=args.nlayer, nhid=args.hidden, \
                                                                 nfeat = num_node_features, nclass = num_graph_labels )
    server = Server(smodel, args.device)

    return clients, server, idx_clients