import random
from random import choices
import numpy as np
import pandas as pd
import sys
import torch
# import metispy as metis

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset

import scipy.sparse as sp
from sklearn import preprocessing


from src.models import *
from src.server import Server,FedGDrop_Server
from src.client import Client_NC, FedGDrop_Client
from src.utils import LargestConnectedComponents, torch_save, torch_load, get_data 

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

def prepareData_oneDS(datapath, data, num_client, batchSize, mode, partition= "METIS", seed=None, overlap=False):

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    if data  == "Cora":
        num_classes, num_features = 7, 1433 
    elif data  == "CiteSeer":
        num_classes, num_features = 6, 3703
    elif data  == "PubMed":
        num_classes, num_features = 3, 500
    elif data  == 'Computers':
        num_classes, num_features = 10, 767
    elif data  == 'Photo':
        num_classes, num_features = 8,745
    else:
        #MS_academic
        raise Exception("MS Academic not yet implemented!")

    
    splitedData = {}
    # df = pd.DataFrame()
    # adj = to_dense_adj(dataset.edge_index)[0]

    for client_id in range(num_client):
        ds = f'{client_id}-{data}'
        #TODO Get Disjoint or Overlapping argument later
        partition = torch_load(datapath, f'{data}_disjoint_{mode}_{partition}/{num_client}/partition_{client_id}.pt')
        #TODO: Check global test data
        global_d = get_data(data, datapath)
        tr, val, tst = partition['client_tr'], partition['client_val'] , partition['client_tst']
        client_num_nodes = tr.x.size(dim=0)
        #Generate dataloaders
        trloader =  DataLoader(dataset= [tr], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)
        valloader =  DataLoader(dataset= [val], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)
        tstloader =  DataLoader(dataset= [tst], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)
        globloader =  DataLoader(dataset= [global_d], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)

        splitedData[ds] = ({'tr': trloader, 'val': valloader, 'tst' : tstloader , 'glob' : globloader}, client_num_nodes)
        #df = get_stats(df, ds, train_data, graphs_val=val_data, graphs_test=test_data)

    return splitedData, num_features, num_classes

def prepareData_fedgdrop_oneDS(datapath, data, num_client, batchSize, mode, partition= "METIS", seed=None, overlap=False):

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    if data  == "Cora":
        num_classes, num_features = 7, 1433 
    elif data  == "CiteSeer":
        num_classes, num_features = 6, 3703
    elif data  == "PubMed":
        num_classes, num_features = 3, 500
    elif data  == 'Computers':
        num_classes, num_features = 10, 767
    elif data  == 'Photo':
        num_classes, num_features = 8,745
    elif data  == 'ogbn-arxiv':
        num_classes, num_features = 40, 128
    else:
        #MS_academic
        raise Exception("MS Academic not yet implemented!")
    
    splitedData = {}

    for client_id in range(num_client):
        ds = f'{client_id}-{data}'
        #TODO Get Disjoint or Overlapping argument later
        if overlap:
            #Right now, we'll have only METIS partition for the overlapping case
            #f'{dataset}_overlapping/{n_comms*n_clien_per_comm}/partition_{comm_id*n_clien_per_comm+client_id}
            part = torch_load(datapath, f'{data}_overlapping/{num_client}/partition_{client_id}.pt')
        else:
            #v2 saves only one client data as we operate over trasnductive setting
            part = torch_load(datapath, f'{data}_disjoint_v2_{partition}/{num_client}/partition_{client_id}.pt')
        #TODO: Check global test data
        cli_graph = part['client_data']
        client_num_nodes = cli_graph.x.size(dim=0)
        #Generate dataloaders

        train_idx = cli_graph.train_mask.nonzero().squeeze(1)
        trloader = torch.utils.data.DataLoader(TensorDataset(train_idx), batch_size= batchSize)

        val_idx =cli_graph.val_mask.nonzero().squeeze(1)
        valloader = torch.utils.data.DataLoader(TensorDataset(val_idx), batch_size= batchSize)

        test_idx = cli_graph.test_mask.nonzero().squeeze(1)
        tstloader = torch.utils.data.DataLoader(TensorDataset(test_idx), batch_size=batchSize)

        adjacency = sp.csr_matrix((np.ones(cli_graph.num_edges, dtype=bool),
                                cli_graph.edge_index),
                                shape=(client_num_nodes, client_num_nodes))

        splitedData[ds] = ({"data": cli_graph, "tr" : trloader , "val": valloader, "tst": tstloader, "adj" : adjacency}, client_num_nodes)
        #df = get_stats(df, ds, train_data, graphs_val=val_data, graphs_test=test_data)

    return splitedData, num_features, num_classes


def setup_devices(splitedData, num_features, num_classes, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloader, train_size = splitedData[ds]
        cmodel_nc = getattr(sys.modules[__name__], args.model)(args.nlayer, num_features, args.hidden, num_classes, args.dropping_method, args.dropout)
        optimizer = torch.optim.Adam(cmodel_nc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_NC(cmodel_nc, idx, ds, dataloader, train_size, optimizer, args))

    #Create server model
    smodel = getattr(sys.modules[__name__], "server"+args.model)(nlayer=args.nlayer, nfeat = num_features,
                                                                 nhid=args.hidden, ncls = num_classes )
    server = Server(smodel, args.device)

    return clients, server, idx_clients

def setup_fedgdrop_devices(splitedData, num_features, num_classes, args):
    idx_clients = {}
    clients = []

    #TODO: Adjust argument
    if args.use_indicators: 
        num_indicators = args.n_hops + 1
    else:
        num_indicators = 0

    #Models have 2 hidden layers for a fair comparison with FedPUB
    for idx, ds in enumerate(splitedData.keys()):
        print("Model creation for client: ", ds, " ...")
        idx_clients[idx] = ds
        dataloader, num_nodes = splitedData[ds]
        cmodel_nc = GCNv2(num_features, hidden_dims=[ args.hidden, args.hidden, num_classes], dropout=args.dropout)
        cmodel_flow = GCNv2(num_features + num_indicators,hidden_dims=[args.hidden, args.hidden, 1])
        opt_nc = torch.optim.Adam(cmodel_nc.parameters(), lr=args.lr)
        log_z = torch.tensor(float(args.log_z_init), requires_grad=True)
        opt_flow = torch.optim.Adam(list(cmodel_flow.parameters()) + [log_z], lr=args.flow_lr)
        
        clients.append(FedGDrop_Client([cmodel_nc,cmodel_flow, log_z], idx, ds, dataloader, num_nodes, num_indicators, [opt_nc, opt_flow], args))

    #Create server model
    smodel_nc = GCNv2(num_features, hidden_dims=[args.hidden,args.hidden, num_classes])
    smodel_flow = GCNv2(num_features + num_indicators,hidden_dims=[args.hidden, args.hidden, 1])

    server = FedGDrop_Server(smodel_nc, smodel_flow, args.device, args.local_flow)

    return clients, server, idx_clients