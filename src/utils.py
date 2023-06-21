import os
import argparse
import pickle
import random
import time
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree, subgraph
import torch.nn.functional as F
from community import community_louvain
import numpy as np
from sklearn.model_selection import train_test_split


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

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


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
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

def nx_louvain_graph_cut(g, num_owners, delta):

    G = to_networkx(g, to_undirected=True)

    node_subjects = g.y

    partition = community_louvain.best_partition(G)

    groups=[]

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
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

    print(groups)

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
    subj_set = [k.item() for k in torch.unique(node_subjects)]
    local_node_subj_0=[]

    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        sbj_i = node_subjects[partition_i]
        local_node_subj_0.append(sbj_i)
    count=[]
    for owner_i in range(num_owners):
        count_i={k:[] for k in subj_set}
        sbj_i=local_node_subj_0[owner_i]
        for i,j in enumerate(sbj_i):
            count_i[j.item()].append(i)
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

    #Get the masks for train validation and training
    train_mask, val_mask , test_mask = g.train_mask, g.val_mask, g.test_mask
    X= g.x
    #Split the graph to num_owners
    for owner_i in range(num_owners):

        local_loaders =  {}

        #Obtain the partition for client i and get the subjects
        partition_i = owner_node_ids[owner_i]

        #Reduce masks over the Louvain partitions
        #CHECK AGAIN
        train_mask_i = [j if i in partition_i else False for i,j in enumerate(train_mask)]
        val_mask_i = [j if i in partition_i else False for i,j in enumerate(val_mask)]
        test_mask_i = [j if i in partition_i else False for i,j in enumerate(test_mask)]

        #Create induced subgraph 
        #CHECK AGAIN
        print(g.edge_index)
        subgraph_i = subgraph(torch.LongTensor(partition_i), g.edge_index)

        local_loaders["train"] = Data(x = X[train_mask_i], y = node_subjects[train_mask_i] ,
                                    edge_index = subgraph_i)

        local_loaders["val"] = Data(x = X[val_mask_i], y = node_subjects[val_mask_i] ,
                                    edge_index = subgraph_i)

        local_loaders["test"] = Data(x = X[test_mask_i], y = node_subjects[test_mask_i] ,
                                    edge_index = subgraph_i)

        local_G.append(local_loaders)

    return local_G