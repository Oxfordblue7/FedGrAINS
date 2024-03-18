import torch
import random
import numpy as np

import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import get_data, split_train, torch_save , louvain_graph_cut

dset= 'ogbn-arxiv'
data_path = '../../../datasets/'
ratio_train = 0.2
seed = 2021
clients = [3, 5, 10, 20]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients, partition, mode = False):
    if mode:
        #Natural split then partition (apply partition to each split) - ours 
        # 20, 70, 10 we apply graph part to 20, 70, 10 separately
        split_subgraphs_v1(n_clients,  get_data(dataset, data_path), dataset, partition)
    else:
        #Manual split then partition - FedPUB
        # i first change from 20/70/10 to 60/2020
        #   then apply graph part based on whole graph and apply split 
        #ratios to each partition 
        #partition the whole graph first then we get 60, 20 20 for each client
        data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
        split_subgraphs_v2(n_clients,data, dataset, partition)

def split_subgraphs_v1(n_clients, data, dataset, partition = "METIS"):
    """
    This function supports Natural splits then Graph partitioning
    This method is Han's suggestion
    """

    tr_graph = data.subgraph(data.train_mask) 
    val_graph = data.subgraph(data.val_mask)
    tst_graph = data.subgraph(data.test_mask)

    G_tr = torch_geometric.utils.to_networkx(tr_graph)
    G_val = torch_geometric.utils.to_networkx(val_graph)
    G_tst = torch_geometric.utils.to_networkx(tst_graph)

    if partition == "METIS":
        #Metis partitioning for each split
        n_cuts_tr, membership_tr = metis.part_graph(G_tr, n_clients)
        n_cuts_val, membership_val = metis.part_graph(G_val, n_clients)
        n_cuts_tst, membership_tst = metis.part_graph(G_tst, n_clients)
    elif partition == "Louvain":
        n_cuts_tr, membership_tr = louvain_graph_cut(G_tr, map(str, tr_graph.y.tolist()), n_clients)
        n_cuts_val, membership_val = louvain_graph_cut(G_val, map(str, val_graph.y.tolist()), n_clients)
        n_cuts_tst, membership_tst = louvain_graph_cut(G_tst, map(str, tst_graph.y.tolist()), n_clients)
    else:
        print("Not implemented yet")

    
    assert len(list(set(membership_tr))) == n_clients
    print(f'Training graph partitioning done,{partition}, n_partitions: {len(list(set(membership_tr)))}, n_lost_edges: {n_cuts_tr}')
    assert len(list(set(membership_val))) == n_clients
    print(f'Validation graph partitioning done,{partition},n_partitions: {len(list(set(membership_val)))}, n_lost_edges: {n_cuts_val}')
    assert len(list(set(membership_tst))) == n_clients
    print(f'Test graph partitioning done,{partition},n_partitions: {len(list(set(membership_tst)))}, n_lost_edges: {n_cuts_tst}')
        
    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        #Fetch client indices and nodes
        if partition == "METIS":    
            client_indices_tr = list(np.where(np.array(membership_tr) == client_id)[0])
            client_indices_val = list(np.where(np.array(membership_val) == client_id)[0])
            client_indices_tst = list(np.where(np.array(membership_tst) == client_id)[0])
        elif partition == "Louvain":
            client_indices_tr = membership_tr[client_id]
            client_indices_val = membership_val[client_id]
            client_indices_tst = membership_tst[client_id]

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

        torch_save(f'{data_path}{dataset}_disjoint_v1_{partition}/{n_clients}/', f'partition_{client_id}.pt', {
            'client_tr': tr_data,
            'client_val': val_data,
            'client_tst': tst_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_tr_nodes: {len(client_indices_tr)}, n_tr_edges: {cli_num_edges_tr}')
        print(f'client_id: {client_id}, iid, n_val_nodes: {len(client_indices_val)}, n_val_edges: {cli_num_edges_val}')
        print(f'client_id: {client_id}, iid, n_test_nodes: {len(client_indices_tst)}, n_test_edges: {cli_num_edges_tst}')

def split_subgraphs_v2(n_clients, data, dataset, partition = "METIS"):

    """
    This function supports manual splits then Graph partitioning case
    Fed-PUB way 
    """
    
    G = torch_geometric.utils.to_networkx(data, to_undirected = True )
    #Graph partitioning
    if partition == "METIS":
        n_cuts, membership = metis.part_graph(G, n_clients)
        assert len(list(set(membership))) == n_clients
    elif partition == "Louvain":
        if n_clients >5:
            n_cuts, membership = louvain_graph_cut(G, list(map(str, data.y.tolist())) , n_clients, delta = 80)
        if n_clients >10:
            n_cuts, membership = louvain_graph_cut(G, list(map(str, data.y.tolist())) , n_clients, delta = 120)
        else:
            n_cuts, membership = louvain_graph_cut(G, list(map(str, data.y.tolist())) , n_clients)
    print(f'graph partition done, {partition}, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')
            
    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        if partition == "METIS":
            client_indices = np.where(np.array(membership) == client_id)[0]
        elif partition =="Louvain":
            client_indices = membership[client_id]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        client_adj = adj[client_indices][:, client_indices]
        client_edge_index, _ = dense_to_sparse(client_adj)
        client_edge_index = client_edge_index.T.tolist()
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]
  
        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index.t().contiguous(),
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        assert torch.sum(client_train_mask).item() > 0

   
        torch_save(f'{data_path}{dataset}_disjoint_v2_{partition}/{n_clients}/', f'partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id} , {partition}, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

for mode in [False]:
    for part in ["METIS"]:      
        for n_clients in clients:
            generate_data(dataset=dset, n_clients=n_clients, partition=part, mode = mode)
