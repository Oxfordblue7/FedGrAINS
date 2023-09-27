import torch
import random
import numpy as np

import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import get_data, split_train, torch_save

data_path = '../../../datasets/'
ratio_train = 0.2
seed = 1234
clients = [5, 10, 20]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients):
    # data = split_train(, dataset, data_path, ratio_train, 'disjoint', n_clients)
    split_subgraphs(n_clients, get_data(dataset, data_path), dataset)

def split_subgraphs(n_clients, data, dataset):

    #Get the natural splits from Cora 
    tr_graph = data.subgraph(data.train_mask) 
    val_graph = data.subgraph(data.val_mask)
    tst_graph = data.subgraph(data.test_mask)

    #Metis partitioning for each split
    G_tr = torch_geometric.utils.to_networkx(tr_graph)
    G_val = torch_geometric.utils.to_networkx(val_graph)
    G_tst = torch_geometric.utils.to_networkx(tst_graph)
    n_cuts_tr, membership_tr = metis.part_graph(G_tr, n_clients)
    n_cuts_val, membership_val = metis.part_graph(G_val, n_clients)
    n_cuts_tst, membership_tst = metis.part_graph(G_tst, n_clients)
    assert len(list(set(membership_tr))) == n_clients
    print(f'Training graph partitioning done, metis, n_partitions: {len(list(set(membership_tr)))}, n_lost_edges: {n_cuts_tr}')
    assert len(list(set(membership_val))) == n_clients
    print(f'Validation graph partitioning done, metis, n_partitions: {len(list(set(membership_val)))}, n_lost_edges: {n_cuts_val}')
    assert len(list(set(membership_tst))) == n_clients
    print(f'Test graph partitioning done, metis, n_partitions: {len(list(set(membership_tst)))}, n_lost_edges: {n_cuts_tst}')
        
    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        #Fetch client indices and nodes
        client_indices_tr = list(np.where(np.array(membership_tr) == client_id)[0])
        client_indices_val = list(np.where(np.array(membership_val) == client_id)[0])
        client_indices_tst = list(np.where(np.array(membership_tst) == client_id)[0])

        #Fetch edge indices and adjacency matrix 
        client_edge_index_tr, client_edge_index_val, client_edge_index_tst = [],[],[]
        client_adj_tr = adj[client_indices_tr][:, client_indices_tr]
        client_adj_val = adj[client_indices_val][:, client_indices_val]
        client_adj_tst = adj[client_indices_tst][:, client_indices_tst]

        client_edge_index_tr = dense_to_sparse(client_adj_tr)[0]
        client_edge_index_val = dense_to_sparse(client_adj_val)[0]
        client_edge_index_tst = dense_to_sparse(client_adj_tst)[0]
        cli_num_edges_tr, cli_num_edges_val = client_edge_index_tr.size(dim=1),client_edge_index_val.size(dim=1)
        cli_num_edges_tst = client_edge_index_tst.size(dim=1)

        #Construct y 
        tr_x,tr_y = data.x[client_indices_tr], data.y[client_indices_tr]
        val_x,val_y = data.x[client_indices_val], data.y[client_indices_val]
        tst_x,tst_y = data.x[client_indices_tst], data.y[client_indices_tst]
  
        tr_data = Data(x = tr_x, y = tr_y,
            edge_index = client_edge_index_tr.contiguous()
        )
        assert tr_x.size(dim=0) > 0
        val_data = Data(x = val_x, y = val_y,
            edge_index = client_edge_index_val.contiguous()
        )
        assert val_x.size(dim=0) > 0
        tst_data = Data(x = tst_x, y = tst_y,
            edge_index = client_edge_index_tst.contiguous()
        )
        assert tst_x.size(dim=0) > 0

        torch_save(f'{data_path}{dataset}_disjoint/{n_clients}/', f'partition_{client_id}.pt', {
            'client_tr': tr_data,
            'client_val': val_data,
            'client_tst': tst_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_tr_nodes: {len(client_indices_tr)}, n_tr_edges: {cli_num_edges_tr}')
        print(f'client_id: {client_id}, iid, n_val_nodes: {len(client_indices_val)}, n_val_edges: {cli_num_edges_val}')
        print(f'client_id: {client_id}, iid, n_test_nodes: {len(client_indices_tst)}, n_test_edges: {cli_num_edges_tst}')

for n_clients in clients:
    generate_data(dataset='Photo', n_clients=n_clients)
