import torch
import wandb
import numpy as np

class Client_NC():
    def __init__(self, model, client_id, name, data_loader, train_size, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = name
        self.train_size = train_size
        self.dataLoader = data_loader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.stats = {'trainingLosses' :[], 'trainingAccs' :[], 'valLosses': [],
                      'valAccs' :[],'testLosses': [] ,'testAccs' :[]}

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
        return eval_nc(self.model, self.dataLoader, "test_mask", self.args.device)

    def local_train_prox(self, local_epoch, mu):
        """ For FedProx """
        tr_loss, tr_acc, val_loss, val_acc = train_nc_prox(self.model, self.id, self.dataLoader, self.optimizer, local_epoch, self.args.device, mu, self.W_old)
        self.stats['trainingLosses'].append(tr_loss)
        self.stats['trainingAccs'].append(tr_acc)
        self.stats['valLosses'].append(val_loss)
        self.stats['valAccs'].append(val_acc)
        

    def evaluate_prox(self, mu):
        return eval_nc_prox(self.model, self.dataLoader, "test_mask", self.args.device, mu, self.W_old)

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
    for epoch in range(local_epoch):
        model.train()
        #Mini-batch loop
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        loss_train, acc_train = eval_nc(model, dataloader, "train_mask", device)
        loss_v, acc_v = eval_nc(model, dataloader, "val_mask", device)

        #After one epoch 
        losses_train.append(loss_train)
        accs_train.append(acc_train)
        losses_val.append(loss_v)
        accs_val.append(acc_v)

        #After local epochs, get the last loss. acc
        wandb.log({f"client-{cli_id}/trainLoss" : losses_train[-1], f"client-{cli_id}/trainAcc" : accs_train[-1] ,
                f"client-{cli_id}/valLoss" : losses_val[-1], f"client-{cli_id}/valAcc" : accs_val[-1]  })

    return losses_train[-1], accs_train[-1], losses_val[-1],accs_val[-1]

@torch.no_grad()
def eval_nc(model, loader, mask, device):
    model.eval()
    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            # TODO: to check the mask is loaded correctly
            loss = model.loss(pred[batch[mask]], label[batch[mask]])
            total_loss = loss.item()
            # TODO: to check the mask is loaded correctly
            preds.append(pred[batch[mask]])
            targets.append(label[batch[mask]])
            lss.append(total_loss)

        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0

        preds = torch.stack(preds).view(targets.size(0) , -1 )
        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc
    
    
def _prox_term(model, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_nc_prox(model, cli_id, dataloader, optimizer, local_epoch, device, mu, Wt):
    losses_train, accs_train, losses_val, accs_val = [], [], [], [] 
    model.train()
    model.to(device)
    for epoch in range(local_epoch):
        model.train()
        
        for batch in dataloader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask]) + mu / 2. * _prox_term(model, Wt)
            loss.backward()
            optimizer.step()
                
        total_loss, acc = eval_nc_prox(model, dataloader, "train_mask", device, mu, Wt)
        loss_v, acc_v = eval_nc_prox(model, dataloader, "val_mask",  device, mu, Wt)

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
def eval_nc_prox(model, loader, mask, device, mu, Wt):
    model.eval()
    model.to(device)

    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            # TODO: to check the mask is loaded correctly
            loss = model.loss(pred[batch[mask]], label[batch[mask]]) + mu / 2. * _prox_term(model, Wt)
            total_loss = loss.item()
            # TODO: to check the mask is loaded correctly
            preds.append(pred[batch[mask]])
            targets.append(label[batch[mask]])
            lss.append(total_loss)
        
        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0
        preds = torch.stack(preds).view(targets.size(0) , -1)
        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc

    