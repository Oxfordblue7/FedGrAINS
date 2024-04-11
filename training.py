import wandb
import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch import Tensor
from collections import defaultdict
from scipy.spatial.distance import cosine

def get_proxy_data(n_feat):
    num_graphs, num_nodes = 5, 100 #n-proxy set as 5
    data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0)) #, seed=self.args.seed
    data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
    return data

#TODO change here based on client
@torch.no_grad()
def get_functional_embedding(data, client):
    client.nc.eval()
    with torch.no_grad():
        proxy_in = data.cuda()
        x, edge_index= proxy_in.x, proxy_in.edge_index
        client.nc.cuda()
        proxy_out  = client.nc(x, edge_index, proxy = True) # , is_proxy=True check here
        proxy_out = proxy_out.mean(dim=0)
        proxy_out = proxy_out.clone().detach().cpu().numpy()
    return proxy_out


def run_selftrain_NC(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    allAccs = {}
    for i,client in enumerate(clients):
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        glob_loss, glob_acc = client.eval_global()
        client.stats['testLosses'].append(loss)
        client.stats['testAccs'].append(acc)
        client.stats['globtestLosses'].append(glob_loss)
        client.stats['globtestAccs'].append(glob_acc)
        wandb.log({f'client-{i}/testLoss' : loss, f'client-{i}/testAcc' : acc})
        wandb.log({f'client-{i}/globtestLoss' : glob_loss, f'client-{i}/globtestAcc' : glob_acc})
        allAccs[client.name] = [max(client.stats['trainingAccs']), max(client.stats['valAccs']),
                                 max(client.stats['testAccs']), max(client.stats['globtestAccs'])]
        print("  > {} done.".format(client.name))

    return allAccs


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch,  samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        print(f"Round {c_round}")
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            np.random.seed(c_round)
            selected_clients = sampling_fn(clients, frac)
        for i,client in enumerate(selected_clients):
            print(f"Client {i}")
            # only get weights of graphconv layers
            client.local_train(local_epoch)
            #Loss is acc for Naive FedGDrop
            #For multiclass . 
            # valLoss, valAcc = client.evaluate(key='val')
            testLoss, testAcc = client.evaluate(key='tst')
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testLoss)

            wandb.log({f'client-{i}/testLoss' : testLoss , f'client-{i}/testAcc' : testAcc})

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    #TODO: LOG AVERAGE OVER CLIENTS
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame

#FedAvg with Personalized aggregation via Functional Embedding Similarities
def run_pfedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, n_feats = 5.0):
    #Initialize the proxy dataset 
    prox_d = get_proxy_data(n_feat = n_feats)

    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        local_embs = [] # Each round, collect local functional embeddings
        print(f"Round {c_round}")
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            np.random.seed(c_round)
            selected_clients = sampling_fn(clients, frac)
        for i,client in enumerate(selected_clients):
            print(f"Client {i}")
            # only get weights of graphconv layers
            client.local_train(local_epoch)
            #Loss is acc for Naive FedGDrop
            #For multiclass . 
            # valLoss, valAcc = client.evaluate(key='val')
            testLoss, testAcc = client.evaluate(key='tst')
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testLoss)
            #Collect the local embedding of ith client
            local_embs.append(get_functional_embedding(prox_d, client))

            wandb.log({f'client-{i}/testLoss' : testLoss , f'client-{i}/testAcc' : testAcc})
        sim_mat = get_sim_matrix(local_embs, frac, len(clients))
        print(sim_mat)
        #Personalized aggregation
        server.aggregate_weights_perv2(selected_clients, sim_mat)


    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    #TODO: LOG AVERAGE OVER CLIENTS
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame

def run_fedpub(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, n_feats = 5.0):
    #Initialize the proxy dataset 
    prox_d = get_proxy_data(n_feat = n_feats)

    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        local_embs = [] # Each round, collect local functional embeddings
        print(f"Round {c_round}")
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            np.random.seed(c_round)
            selected_clients = sampling_fn(clients, frac)
        for i,client in enumerate(selected_clients):
            print(f"Client {i}")
            # only get weights of graphconv layers
            client.local_masked_train(local_epoch)
            #Loss is acc for Naive FedGDrop
            #For multiclass . 
            # valLoss, valAcc = client.evaluate(key='val')
            testLoss, testAcc = client.evaluate(key='tst')
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testLoss)
            #Collect the local embedding of ith client
            local_embs.append(get_functional_embedding(prox_d, client))

            wandb.log({f'client-{i}/testLoss' : testLoss , f'client-{i}/testAcc' : testAcc})
        sim_mat = get_sim_matrix(local_embs, frac, len(clients))
        print(sim_mat)
        #Personalized aggregation
        server.aggregate_weights_perv2(selected_clients, sim_mat)


    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    #TODO: LOG AVERAGE OVER CLIENTS
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame



def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, mu, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            np.random.seed(c_round)
            selected_clients = sampling_fn(clients, frac)

        for i,client in enumerate(selected_clients):
            #TODO assume full participation
            client.local_train_prox(local_epoch, mu)
            testLoss, testAcc = client.evaluate_prox(mu)
            globtestLoss, globtestAcc = client.eval_global_prox(mu)
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testLoss) 
            client.stats['globtestLosses'].append(globtestLoss)
            client.stats['globtestAccs'].append(globtestLoss)
            wandb.log({f'client-{i}/testLoss' : testLoss, f'client-{i}/testAcc' : testAcc})
            wandb.log({f'client-{i}/globtestLoss' : globtestLoss, f'client-{i}/globtestAcc' : globtestAcc})

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

            # cache the aggregated weights for next round
            client.cache_weights()

    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]


    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame

def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1 = 0.05, EPS_2 = 0.1):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

        #Based on classification models (not based on GFlowNets)
        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, clients[idc[0]].W_nc, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters, agg_flow = True)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W_nc, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)

    return frame

#Only node classifiers clustered. we apply Fedavg on FlowNets
def run_gcfl_nc(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1 = 0.05, EPS_2 = 0.1):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

        #Based on classification models (not based on GFlowNets)
        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, clients[idc[0]].W_nc, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        
        server.aggregate_clusterwise(client_clusters, agg_flow = False)
        server.aggregate_flownets(participating_clients) #Fedavg of clients
        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W_nc, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)
    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(frame)

    return frame

def run_gcflplus(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1 = 0.05, EPS_2 = 0.1, seq_length = 5, standardize = False):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()
            

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W_nc, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]
        # frame.loc[client.name, 'globtest_acc'] = client.stats['globtestAccs'][client.stats['valAccs'].index(m)]
    mean_frame = frame.mean(axis=0)
    wandb.log({'trainAcc': mean_frame['train_acc'] ,'valAcc' : mean_frame['val_acc'] , 'testAcc' :  mean_frame['test_acc']})

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)

    return frame



def get_sim_matrix(local_f_embs, n_frac , n_clients):
    n_connected = round(n_clients * n_frac)
    assert n_connected == len(local_f_embs)
    sim_matrix = np.empty(shape=(n_connected, n_connected))
    for i in range(n_connected):
        for j in range(n_connected):
            sim_matrix[i, j] = 1 - cosine(local_f_embs[i], local_f_embs[j])
    #self.args.norm_scale
    sim_matrix = np.exp(10 * sim_matrix)
        
    #Row normalize matrix such that each row is softmax
    row_sums = sim_matrix.sum(axis=1)
    sim_matrix = sim_matrix / row_sums[:, np.newaxis]
    return torch.tensor(sim_matrix)

def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data
