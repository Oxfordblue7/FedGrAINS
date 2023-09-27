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
from sklearn import preprocessing
from torch_geometric.utils import to_networkx, to_dense_adj, dense_to_sparse


from src.models import *
from src.server import Server
from src.client import Client_NC
from src.utils import LargestConnectedComponents, torch_save, torch_load  

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

# TODO: modify to load saved partitioned data directly
# TODO: Create a separate data generation script to generate partitions
def prepareData_oneDS(datapath, data, num_client, batchSize, seed=None, overlap=False):

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

    # #Train-val-test split
    # #Train ratio is 0.2 in accordance with FedPUB
    # dataset = _split_train(dataset)

    # #Disjoint metis partitioning 
    # _ , membership =  _metis_graph_cut(dataset, num_client)
    
    splitedData = {}
    # df = pd.DataFrame()
    # adj = to_dense_adj(dataset.edge_index)[0]

    for client_id in range(num_client):
        ds = f'{client_id}-{data}'
        #TODO Get Disjoint or Overlapping argument later
        partition = torch_load(datapath, f'{data}_disjoint/{num_client}/partition_{client_id}.pt')
        #Load global test
        
        tr, val, tst = partition['client_tr'], partition['client_val'] , partition['client_tst']
        client_num_nodes = tr.x.size(dim=0)
        #Generate dataloaders
        #Couldnt run Cora with mini-batching
        trloader =  DataLoader(dataset= [tr], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)
        valloader =  DataLoader(dataset= [val], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)
        tstloader =  DataLoader(dataset= [tst], batch_size=batchSize, 
                                 shuffle=False, pin_memory=False)

        splitedData[ds] = ({'tr': trloader, 'val': valloader, 'tst' : tstloader}, client_num_nodes)
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