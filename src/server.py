import os
import torch
import numpy as np
import random
import networkx as nx

class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data,\
                                                                         client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_per(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            if 'graph_convs' in k:
                self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]



class FedGDrop_Server():
    def __init__(self, nc, flow, device, local_flow = True):
        self.nc = nc.to(device)
        self.flow = flow.to(device)
        self.W_nc = {key: value for key, value in self.nc.named_parameters()}
        self.model_cache = []
        self.local_flow = local_flow
        self.W_flow = None if local_flow else {key: value for key, value in self.flow.named_parameters()}

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W_nc.keys():
                dW[k] = client.dW_nc[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.num_nodes
        for k in self.W_nc.keys():
            self.W_nc[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_nc[k].data,\
                                                                         client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()
        if self.W_flow is not None:
            print("FedAvg for flow model")
            for k in self.W_flow.keys():
                self.W_flow[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_flow[k].data,\
                                                                            client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()
                
    def aggregate_flownets(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        print("FedAvg for flow model")
        for client in selected_clients:
            total_size += client.num_nodes
            for k in self.W_flow.keys():
                self.W_flow[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_flow[k].data,\
                                                                            client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_per(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.num_nodes
        for k in self.W_nc.keys():
            if 'gcn_layers' in k:
                self.W_nc[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_nc[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        if self.W_flow is not None:
            for k in self.W_flow.keys():
                if 'gcn_layers' in k:
                    self.W_flow[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_flow[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def aggregate_weights_perv2(self, selected_clients, sim_mat):
        # pass train_size, and weighted aggregate
        total_size = 0
        for j, client in enumerate(selected_clients):
            total_size += client.num_nodes
            for k in client.W_nc.keys():
                if 'gcn_layers' in k:
                    client.W_nc[k].data = torch.sum(torch.stack([torch.mul(client.W_nc[k].data, sim_mat[j,i]) for i,client in enumerate(selected_clients)]), dim=0).clone()
        
        if self.W_flow is not None:
            print("FedAvg for flow model")
            for k in self.W_flow.keys():
                self.W_flow[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W_flow[k].data,\
                                                                            client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W_nc.keys():
                dW[k] = client.dW_nc[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def aggregate_clusterwise(self, client_clusters, agg_flow = True):
        for cluster in client_clusters:
            targs_nc, targs_flow = [] , []
            sours_nc, sours_flow = [] , []
            total_size_nc, total_size_flow = 0 , 0
            for client in cluster:
                W_nc, W_flow = {} , {}
                dW_nc, dW_flow = {} , {}
                for k in self.W_nc.keys():
                    W_nc[k] = client.W_nc[k]
                    dW_nc[k] = client.dW_nc[k]
                if agg_flow:
                    if self.W_flow is not None:
                        for k in self.W_flow.keys():
                            W_flow[k] = client.W_flow[k]
                            dW_flow[k] = client.dW_flow[k]
                targs_nc.append(W_nc)
                sours_nc.append((dW_nc, client.num_nodes))
                total_size_nc += client.num_nodes
                if agg_flow:
                    targs_flow.append(W_flow)
                    sours_flow.append((dW_flow, client.num_nodes))
                    total_size_flow += client.num_nodes

            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs_nc, sources=sours_nc, total_size=total_size_nc)
            reduce_add_average(targets=targs_flow, sources=sours_flow, total_size=total_size_flow)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W_nc.keys():
                dW[k] = client.dW_nc[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]
        
    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()


def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp.to(target[name].data.device)


def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)