import random
from random import choices
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data,DataLoader
from sklearn import preprocessing
from torch_geometric.utils import to_networkx, subgraph

from community import community_louvain

from src.models import *
from src.server import Server
from src.client import Client_GC
from src.utils import get_maxDegree, get_stats, split_data, get_numGraphLabels


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

        local_loaders =  {}

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
        pyg_dataset = Planetoid(f"{datapath}/Planetoid", data)
        num_classes= pyg_dataset.num_classes
        num_features  = pyg_dataset.num_features
        dataset = pyg_dataset[0]
    else:
        #MS_academic
        raise Exception("MS Academic not yet implemented!")


    #Louvain partitioning the graph
    graphs_chunks = _louvain_graph_cut(pyg_dataset, num_client, delta)
    #graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    #num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        #Train-val-tst split
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
        #Generate dataloaders
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, num_classes, df


def setup_devices(splitedData, num_classes, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = getattr(sys.modules[__name__], args.model)(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dm)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    #Create server model
    smodel = getattr(sys.modules[__name__], "server"+args.model)(nlayer=args.nlayer, nhid=args.hidden , nout = num_classes)
    server = Server(smodel, args.device)

    return clients, server, idx_clients