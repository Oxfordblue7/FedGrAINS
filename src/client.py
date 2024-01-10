from tqdm import tqdm
import torch
import wandb
import argparse
import numpy as np
import torch_geometric
from torch.distributions import Bernoulli, Gumbel
from sklearn.metrics import accuracy_score, f1_score
from .utils import get_neighborhoods, sample_neighborhoods_from_probs, slice_adjacency, TensorMap

class Client_NC():
    def __init__(self, model, client_id, name, data_loader, train_size, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = name
        self.train_size = train_size
        self.dataLoader = data_loader #This is a dict
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.stats = {'trainingLosses' :[], 'trainingAccs' :[], 'valLosses': [],
                      'valAccs' :[],'testLosses': [] ,'testAccs' :[],
                      'globtestLosses': [] ,'globtestAccs' :[]}

    def download_from_server(self, server):
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.W.keys())

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        tr_loss, tr_acc, val_loss, val_acc =  train_nc(self.model, self.id, self.dataLoader, self.optimizer, local_epoch, self.args.device)
        self.stats['trainingLosses'].append(tr_loss)
        self.stats['trainingAccs'].append(tr_acc)
        self.stats['valLosses'].append(val_loss)
        self.stats['valAccs'].append(val_acc)

    def compute_weight_update(self, local_epoch):
        """ For GCFL """
        copy(target=self.W_old, source=self.W, keys=self.W.keys())

        self.train_stats = train_nc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        _subtract(target=self.dW, minuend=self.W, subtrahend=self.W_old)

    def evaluate(self):
        return eval_nc(self.model, self.dataLoader['tst'], self.args.device)
    def eval_global(self):
        return eval_nc(self.model, self.dataLoader['glob'], self.args.device)

    def local_train_prox(self, local_epoch, mu):
        """ For FedProx """
        tr_loss, tr_acc, val_loss, val_acc = train_nc_prox(self.model, self.id, self.dataLoader, self.optimizer, local_epoch, self.args.device, mu, self.W_old)
        self.stats['trainingLosses'].append(tr_loss)
        self.stats['trainingAccs'].append(tr_acc)
        self.stats['valLosses'].append(val_loss)
        self.stats['valAccs'].append(val_acc)
        
    def evaluate_prox(self, mu):
        return eval_nc_prox(self.model, self.dataLoader['tst'], self.args.device, mu, self.W_old)
    def eval_global_prox(self, mu):
        return eval_nc_prox(self.model, self.dataLoader['glob'], self.args.device, mu, self.W_old)


class FedGDrop_Client():
    def __init__(self, model, client_id, name, data_loader, num_nodes, num_indicators, optimizer, args):
        self.nc = model[0].to(args.device)
        self.flow = model[1].to(args.device)
        self.logz = model[2].to(args.device)
        self.id = client_id
        self.name = name
        self.num_nodes = num_nodes
        self.num_ind = num_indicators
        self.dataLoader = data_loader #This is a dict
        self.opt_nc = optimizer[0]
        self.opt_flow = optimizer[1]
        self.args = args

        self.W_nc = {key: value for key, value in self.nc.named_parameters()}
        self.dW_nc = {key: torch.zeros_like(value) for key, value in self.nc.named_parameters()}
        self.W_nc_old = {key: value.data.clone() for key, value in self.nc.named_parameters()}

        self.W_flow = {key: value for key, value in self.flow.named_parameters()}
        self.dW_flow = {key: torch.zeros_like(value) for key, value in self.flow.named_parameters()}
        self.W_flow_old = {key: value.data.clone() for key, value in self.flow.named_parameters()}

        self.node_map = TensorMap(size=num_nodes)

        self.stats = {'trainingLosses' :[], 'trainingAccs' :[], 'valLosses': [],
                      'valAccs' :[],'testLosses': [] ,'testAccs' :[],
                      'globtestLosses': [] ,'globtestAccs' :[]}

    def download_from_server(self, server):
        for k in server.W_nc:
            self.W_nc[k].data = server.W_nc[k].data.clone()
        for k in server.W_flow:
            self.W_flow[k].data = server.W_flow[k].data.clone()

    def cache_weights(self):
        for name in self.W_nc.keys():
            self.W_nc_old[name].data = self.W_nc[name].data.clone()
        for name in self.W_flow.keys():
            self.W_flow_old[name].data = self.W_flow[name].data.clone()

    def reset(self):
        copy(target=self.W_nc, source=self.W_nc_old, keys=self.W_nc.keys())
        copy(target=self.W_flow, source=self.W_flow_old, keys=self.W_flow.keys())

    def local_train(self, local_epoch):
        """ FedGDrop training for each client"""
        tr_loss, tr_acc, val_loss, val_acc =  train_fedgdrop_nc(self.nc, self.flow, self.logz, self.id, self.dataLoader, 
                                                                self.opt_nc, self.opt_flow, self.num_nodes, self.num_ind,
                                                                self.node_map, local_epoch, self.args.device, self.args)
        self.stats['trainingLosses'].append(tr_loss)
        self.stats['trainingAccs'].append(tr_acc)
        self.stats['valLosses'].append(val_loss)
        self.stats['valAccs'].append(val_acc)

    def evaluate(self):
        return eval_fedgdrop(self.nc,self.flow, self.dataLoader['data'], self.args,
                             self.dataLoader['adj'], self.node_map, self.num_ind, self.args.device,
                             self.dataLoader['data'].test_mask,self.args.eval_on_cpu,loader= self.dataLoader['tst'],
                            full_batch=self.args.eval_full_batch)
    def eval_global(self):
        return eval_fedgdrop(self.nc,self.flow, self.dataLoader['data'], self.args,
                             self.dataLoader['adj'], self.node_map, self.num_ind, self.args.device,
                             self.dataLoader['data'].test_mask,self.args.eval_on_cpu,loader= self.dataLoader['tst'],
                            full_batch=self.args.eval_full_batch)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def _subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_nc(model, cli_id, dataloader, optimizer, local_epoch, device):
    losses_train, accs_train, losses_val, accs_val = [], [], [], [] 
    model.to(device)
    tr_loader, val_loader = dataloader['tr'] , dataloader['val']
    for epoch in range(local_epoch):
        model.train()
        #Mini-batch loop
        for i, batch in enumerate(tr_loader):
            optimizer.zero_grad()
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            loss = model.loss(pred, batch.y)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = eval_nc(model, tr_loader, device)
        loss_v, acc_v = eval_nc(model, val_loader, device)

        #After one epoch 
        losses_train.append(loss_train)
        accs_train.append(acc_train)
        losses_val.append(loss_v)
        accs_val.append(acc_v)

        #After local epochs, get the last loss. acc
        wandb.log({f"client-{cli_id}/trainLoss" : losses_train[-1], f"client-{cli_id}/trainAcc" : accs_train[-1] ,
                f"client-{cli_id}/valLoss" : losses_val[-1], f"client-{cli_id}/valAcc" : accs_val[-1]  })

    return losses_train[-1], accs_train[-1], losses_val[-1],accs_val[-1]

def train_fedgdrop_nc(nc, flow, log_z, cli_id, dataloader, opt_nc, opt_flow, num_nodes, num_ind, node_map, local_epoch, device, args):

    #TODO: Delete some vars later for memory
    nc.to(device)
    flow.to(device)
    data = dataloader['data']
    train_loader = dataloader['tr']
    if data.y.dim() == 1:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    prev_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((num_nodes, num_ind))

    adjacency = dataloader['adj']

    # logger.info('Training')
    for epoch in range(1, local_epoch + 1):
        acc_loss_gfn = 0
        acc_loss_c = 0
        # add a list to collect memory usage


        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True

                indicator_features.zero_()
                indicator_features[target_nodes, -1] = 1.0

                global_edge_indices = []
                log_probs = []
                sampled_sizes = []
                neighborhood_sizes = []
                all_statistics = []
                # Sample neighborhoods with the GCN-GF model: 2-Hop Sampling
                for hop in range(args.n_hops):
                    # Get neighborhoods of target nodes in batch
                    neighborhoods = get_neighborhoods(previous_nodes, adjacency)

                    # Identify batch nodes (nodes + neighbors) and neighbors
                    prev_nodes_mask.zero_()
                    batch_nodes_mask.zero_()
                    prev_nodes_mask[previous_nodes] = True
                    batch_nodes_mask[neighborhoods.view(-1)] = True
                    neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

                    batch_nodes = node_map.values[batch_nodes_mask]
                    neighbor_nodes = node_map.values[neighbor_nodes_mask]
                    indicator_features[neighbor_nodes, hop] = 1.0

                    # Map neighborhoods to local node IDs
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)
                    # Select only the needed rows from the feature and
                    # indicator matrices
                    if args.use_indicators:
                        x = torch.cat([data.x[batch_nodes],
                                    indicator_features[batch_nodes]],
                                    dim=1
                                    ).to(device)
                    else:
                        x = data.x[batch_nodes].to(device)

                    # Get probabilities for sampling each node
                    node_logits, _ = flow(x, local_neighborhoods)
                    # Select logits for neighbor nodes only
                    node_logits = node_logits[node_map.map(neighbor_nodes)]
                    # Sample neighbors using the logits
                    # num_samples = 16
                    #TODO : ASSERT num_samples <= batch_size
                    sampled_neighboring_nodes, log_prob, statistics = sample_neighborhoods_from_probs(
                        node_logits,
                        neighbor_nodes,
                        args.k
                    )
                    all_nodes_mask[sampled_neighboring_nodes] = True
                    log_probs.append(log_prob)
                    sampled_sizes.append(sampled_neighboring_nodes.shape[0])
                    neighborhood_sizes.append(neighborhoods.shape[-1])
                    all_statistics.append(statistics)

                    # Update batch nodes for next hop
                    batch_nodes = torch.cat([target_nodes,
                                             sampled_neighboring_nodes],
                                            dim=0)

                    # Retrieve the edge index that results after sampling
                    k_hop_edges = slice_adjacency(adjacency,
                                                  rows=previous_nodes,
                                                  cols=batch_nodes)
                    global_edge_indices.append(k_hop_edges)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]

                #TODO: There is a bug in copying models. At some point, nc goes to cpu
                #Somehow solved with this, but still we should check it .
                x = data.x[all_nodes].to(device)
                logits, gcn_mem_alloc = nc(x, edge_indices)

                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits[local_target_ids],
                                 data.y[target_nodes].to(device)) + args.reg_param *torch.sum(torch.var(logits, dim=1))

                opt_nc.zero_grad()

                loss_c.backward()

                opt_nc.step()

                opt_flow.zero_grad()
                cost_gfn = loss_c.detach()
                #loss_coef: float = 1e4
                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + args.loss_coef*cost_gfn)**2

                loss_gfn.backward()

                opt_flow.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

                wandb.log({f'client-{cli_id}/batch_loss_gfn': batch_loss_gfn,
                           f'client-{cli_id}/batch_loss_c': batch_loss_c,
                           f'client-{cli_id}/log_z': log_z,
                           f'client-{cli_id}/-log_probs': -torch.sum(torch.cat(log_probs, dim=0))})

                log_dict = {}
                for i, statistics in enumerate(all_statistics):
                    for key, value in statistics.items():
                        log_dict[f"{key}_{i}"] = value
                wandb.log(log_dict)

                acc_loss_gfn += batch_loss_gfn / len(train_loader)
                acc_loss_c += batch_loss_c / len(train_loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                 'batch_loss_c': batch_loss_c,
                                 'log_z': log_z.item(),
                                 'log_probs': torch.sum(torch.cat(log_probs, dim=0)).item()})
                bar.update()

        
        bar.close()

        val_accuracy, val_f1 = eval_fedgdrop(nc,
                                      flow,
                                      data,
                                      args,
                                      adjacency,
                                      node_map,
                                      num_ind,
                                      device,
                                      data.val_mask,
                                      args.eval_on_cpu,
                                      loader=dataloader['val'],
                                      full_batch=args.eval_full_batch)

        tr_accuracy, tr_f1 = eval_fedgdrop(nc,
                                      flow,
                                      data,
                                      args,
                                      adjacency,
                                      node_map,
                                      num_ind,
                                      device,
                                      data.train_mask,
                                      args.eval_on_cpu,
                                      loader=dataloader['tr'],
                                      full_batch=args.eval_full_batch)
        
    return tr_accuracy, tr_f1, val_accuracy, val_f1


@torch.no_grad()
def eval_nc(model, loader, device):
    model.eval()
    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            # TODO: to check the mask is loaded correctly (no need for mask rn)
            loss = model.loss(pred, label)
            total_loss = loss.item()
            preds.append(pred)
            targets.append(label)
            lss.append(total_loss)

        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0

        preds = torch.stack(preds).view(targets.size(0) , -1 )
        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc
    

@torch.inference_mode()
def eval_fedgdrop(gcn_c: torch.nn.Module,
             gcn_gf: torch.nn.Module,
             data: torch_geometric.data.Data,
             args: argparse.Namespace,
             adjacency: torch.Tensor,
             node_map: TensorMap,
             num_indicators: int,
             device: torch.device,
             mask: torch.Tensor = None,
             eval_on_cpu: bool = True,
             loader: torch.utils.data.DataLoader = None,
             full_batch: bool = False,
             ) -> tuple[float, float]:
    """
    Evaluate the model on the validation or test set. This can be done in two ways: either by performing full-batch
    message passing or by performing mini-batch message passing. The latter is more memory efficient, but the former is
    faster.
    """

    x = data.x
    edge_index = data.edge_index
    if eval_on_cpu:
        # move data to CPU
        x = x.cpu()
        edge_index = edge_index.cpu()
        gcn_c = gcn_c.cpu()
        all_predictions = torch.tensor([], dtype=torch.long, device='cpu')
    else:
        # move data to GPU
        x = x.to(device)
        edge_index = edge_index.to(device)
        gcn_c = gcn_c.to(device)
        all_predictions = torch.tensor([], dtype=torch.long, device='cuda')

    if full_batch:
        # perform full batch message passing for evaluation
        logits_total, _ = gcn_c(x, edge_index)
        if data.y[mask].dim() == 1:
            predictions = torch.argmax(logits_total, dim=1)[mask].cpu()
            targets = data.y[mask]
            accuracy = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions, average='micro')
        # multilabel classification
        else:
            y_pred = logits_total[mask] > 0
            y_true = data.y[mask] > 0.5

            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = accuracy = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.
    else:
        # perform mini-batch message passing for evaluation
        assert loader is not None, 'loader must be provided if full_batch is False'

        prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        indicator_features = torch.zeros((data.num_nodes, num_indicators))

        for batch_id, batch in enumerate(loader):
            target_nodes = batch[0]

            previous_nodes = target_nodes.clone()
            all_nodes_mask = torch.zeros_like(prev_nodes_mask)
            all_nodes_mask[target_nodes] = True

            indicator_features.zero_()
            indicator_features[target_nodes, -1] = 1.0

            global_edge_indices = []

            # Sample neighborhoods with the GCN-GF model
            for hop in range(args.sampling_hops):
                # Get neighborhoods of target nodes in batch
                neighborhoods = get_neighborhoods(previous_nodes, adjacency)

                # Identify batch nodes (nodes + neighbors) and neighbors
                prev_nodes_mask.zero_()
                batch_nodes_mask.zero_()
                prev_nodes_mask[previous_nodes] = True
                batch_nodes_mask[neighborhoods.view(-1)] = True
                neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

                batch_nodes = node_map.values[batch_nodes_mask]
                neighbor_nodes = node_map.values[neighbor_nodes_mask]
                indicator_features[neighbor_nodes, hop] = 1.0

                # Map neighborhoods to local node IDs
                node_map.update(batch_nodes)
                local_neighborhoods = node_map.map(neighborhoods).to(device)
                # Select only the needed rows from the feature and
                # indicator matrices
                if args.use_indicators:
                    x = torch.cat([data.x[batch_nodes],
                                   indicator_features[batch_nodes]],
                                  dim=1
                                  ).to(device)
                else:
                    x = data.x[batch_nodes].to(device)

                # Get probabilities for sampling each node
                node_logits, _ = gcn_gf(x, local_neighborhoods)
                # Select logits for neighbor nodes only
                node_logits = node_logits[node_map.map(neighbor_nodes)]
                # Sample top k neighbors using the logits
                b = Bernoulli(logits=node_logits.squeeze())
                samples = torch.topk(b.probs, k=args.num_samples, dim=0, sorted=False)[1]
                sample_mask = torch.zeros_like(node_logits.squeeze(), dtype=torch.float)
                sample_mask[samples] = 1
                sampled_neighboring_nodes = neighbor_nodes[sample_mask.bool().cpu()]

                all_nodes_mask[sampled_neighboring_nodes] = True

                # Update batch nodes for next hop
                batch_nodes = torch.cat([target_nodes,
                                         sampled_neighboring_nodes],
                                        dim=0)

                # Retrieve the edge index that results after sampling
                k_hop_edges = slice_adjacency(adjacency,
                                              rows=previous_nodes,
                                              cols=batch_nodes)
                global_edge_indices.append(k_hop_edges)

                # Update the previous_nodes
                previous_nodes = batch_nodes.clone()

            all_nodes = node_map.values[all_nodes_mask]
            node_map.update(all_nodes)
            edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]

            x = data.x[all_nodes].to(device)
            logits_total, _ = gcn_c(x, edge_indices)
            predictions = torch.argmax(logits_total, dim=1)
            predictions = predictions[node_map.map(target_nodes)]  # map back to original node IDs

            all_predictions = torch.cat([all_predictions, predictions], dim=0)

        all_predictions = all_predictions.cpu()
        targets = data.y[mask]

        accuracy = accuracy_score(targets, all_predictions)
        f1 = f1_score(targets, all_predictions, average='micro')

    return accuracy, f1


def _prox_term(model, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_nc_prox(model, cli_id, dataloader, optimizer, local_epoch, device, mu, Wt):
    losses_train, accs_train, losses_val, accs_val = [], [], [], [] 
    model.train()
    model.to(device)
    tr_loader, val_loader = dataloader['tr'] , dataloader['val']
    for epoch in range(local_epoch):
        model.train()
        
        for batch in tr_loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = model.loss(pred, batch.y) + mu / 2. * _prox_term(model, Wt)
            loss.backward()
            optimizer.step()
                
        total_loss, acc = eval_nc_prox(model, tr_loader, device, mu, Wt)
        loss_v, acc_v = eval_nc_prox(model, val_loader, device, mu, Wt)

        #After one epoch (for round 1 )
        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
    
        #After local epochs, get the last loss. acc
        wandb.log({f"client-{cli_id}/trainLoss" : losses_train[-1], f"client-{cli_id}/trainAcc" : accs_train[-1] ,
                f"client-{cli_id}/valLoss" : losses_val[-1], f"client-{cli_id}/valAcc" : accs_val[-1]  })

    return losses_train[-1], accs_train[-1], losses_val[-1],accs_val[-1]

@torch.no_grad()
def eval_nc_prox(model, loader,  device, mu, Wt):
    model.eval()
    model.to(device)

    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            # TODO: to check the mask is loaded correctly
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, Wt)
            total_loss = loss.item()
            # TODO: to check the mask is loaded correctly
            preds.append(pred)
            targets.append(label)
            lss.append(total_loss)
        
        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0
        preds = torch.stack(preds).view(targets.size(0) , -1)
        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc

    