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
from torch_geometric.utils import to_networkx, subgraph, to_dense_adj, dense_to_sparse


from community import community_louvain

from src.models import *
from src.server import Server
from src.client import Client_GC
from src.utils import LargestConnectedComponents, get_maxDegree, get_stats, split_data, get_numGraphLabels

def _metis_graph_cut(g, num_owners):
    G = to_networkx(g)
    n_cuts, membership = metis.part_graph(G, num_owners)
    assert len(list(set(membership))) == num_owners
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')
    return n_cuts, membership    

def _louvain_graph_cut(g, num_owners, delta):

    G = to_networkx(g[0], to_undirected=True)
    node_features = g.x
    node_subjects = g.y

    partition = community_louvain.best_partition(G)

    groups=[]

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    #print(groups)
    partition_groups = {group_i:[] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max=len(list(G.nodes()))//num_owners-delta
    for group_i in groups:
        while len(partition_groups[group_i])>group_len_max:
            long_group=list.copy(partition_groups[group_i])
            partition_groups[group_i]=list.copy(long_group[:group_len_max])
            new_grp_i=max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i]=long_group[group_len_max:]
    #print(partition_groups)

    len_list=[]
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
    len_dict={}

    for i in range(len(groups)):
        len_dict[groups[i]]=len_list[i]
    sort_len_dict={k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1],reverse=True)}

    owner_node_ids={owner_id:[] for owner_id in range(num_owners)}

    owner_nodes_len=len(list(G.nodes()))//num_owners
    owner_list=[i for i in range(num_owners)]
    owner_ind=0

    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]]+=partition_groups[group_i]
    for owner_i in owner_node_ids.keys():
        print('nodes len for '+str(owner_i)+' = '+str(len(owner_node_ids[owner_i])))

    local_G = []
    local_node_subj = []
    local_nodes_ids = []
    target_encoding = preprocessing.LabelBinarizer()
    target = target_encoding.fit_transform(node_subjects)
    local_target = []
    subj_set = [k.item() for k in torch.unique(node_subjects)]
    local_node_subj_0=[]

    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        #locs_i = g[0][partition_i]
        sbj_i = node_subjects.clone()
        sbj_i[:] = "" if node_subjects[0].__class__ == str else 0
        sbj_i[partition_i] = node_subjects[partition_i]
        local_node_subj_0.append(sbj_i)
    count=[]
    for owner_i in range(num_owners):
        count_i={k:[] for k in subj_set}
        sbj_i=local_node_subj_0[owner_i]
        for i,j in enumerate(sbj_i):
            if j!=0 and j!="":
                count_i[int(j.item())].append(i)
        count.append(count_i)
    for k in subj_set:
        for owner_i in range(num_owners):
            if len(count[owner_i][k])<2:
                for j in range(num_owners):
                    if len(count[j][k])>2:
                        id=count[j][k][-1]
                        count[j][k].remove(id)
                        count[owner_i][k].append(id)
                        owner_node_ids[owner_i].append(id)
                        owner_node_ids[j].remove(id)
                        j=num_owners

    #Split the graph to num_owners
    for owner_i in range(num_owners):

        #Obtain the partition for client i and get the subjects
        partition_i = owner_node_ids[owner_i]
        #Create induced subgraph 
        subgraph_i = subgraph(torch.LongTensor(partition_i), g.edge_index)[0]
        G_i = g[0].subgraph(subgraph_i)
        print("Data of owner ", owner_i , " " , G_i)
        
        local_G.append(G_i)

    # return local_G

    return local_G

def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, delta, batchSize, convert_x=False, seed=None, overlap=False):
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
    n_data = dataset.num_nodes
    ratio_test = (1-0.2)/2
    n_train = round(n_data * 0.2)
    n_test = round(n_data * ratio_test)
    
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    dataset.train_mask.fill_(False)
    dataset.test_mask.fill_(False)
    dataset.val_mask.fill_(False)

    dataset.train_mask[train_indices] = True
    dataset.test_mask[test_indices] = True
    dataset.val_mask[val_indices] = True

    #Louvain partitioning the graph
    #graphs_chunks = _louvain_graph_cut(pyg_dataset, num_client, delta)
    #graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    #Disjoint metis partitioning
    n_cuts, membership =  _metis_graph_cut(dataset, num_client)
    
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
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

        #Generate dataloaders
        #Couldnt run Cora with mini-batching
        dataloader =  DataLoader([client_data], batch_size=batchSize, shuffle=False)
        splitedData[ds] = (dataloader,num_features, num_classes, len(client_data))
        #df = get_stats(df, ds, train_data, graphs_val=val_data, graphs_test=test_data)

    return splitedData, num_classes


def setup_devices(splitedData, num_classes, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloader, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = getattr(sys.modules[__name__], args.model)(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropping_method)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloader, optimizer, args))

    #Create server model
    smodel = getattr(sys.modules[__name__], "server"+args.model)(nlayer=args.nlayer, nhid=args.hidden )
    server = Server(smodel, args.device)

    return clients, server, idx_clients