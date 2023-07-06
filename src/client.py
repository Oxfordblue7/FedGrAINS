import torch
import wandb
import numpy as np

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.dropout = args.dropout
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
                self.W[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_gc(self.model, self.name, self.dataLoader, self.dropout, self.optimizer, local_epoch, self.args.device)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update(self, local_epoch):
        """ For GCFL """
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self):
        return eval_gc(self.model, self.dataLoader, self.dataLoader.dataset[0].test_mask,  self.args.device)

    def local_train_prox(self, local_epoch, mu):
        """ For FedProx """
        train_stats = train_gc_prox(self.model, self.dataLoader, self.dropout, self.optimizer, local_epoch, self.args.device,
                               self.gconvNames, self.W, mu, self.W_old)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate_prox(self, mu):
        return eval_gc_prox(self.model, self.dataLoader['test'], self.args.device, self.gconvNames, mu, self.W_old)

def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()



def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_gc(model, name, dataloader, dropout, optimizer, local_epoch, device):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    model.train()
    model.to(device)
    for epoch in range(local_epoch):
        model.train()
    for _,batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch.to(device)
        pred = model(batch.x, batch.edge_index, dropout)
        loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        
            
        total_loss, acc = eval_gc(model, dataloader, batch.train_mask,  device)
        loss_v, acc_v = eval_gc(model, dataloader,batch.val_mask , device)
        loss_tt, acc_tt = eval_gc(model, dataloader, batch.test_mask, device)

        #After one epoch 
        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    #After local epochs, get the averaged loss. acc
    # wandb.log({'{}-train_loss'.format(name)  : np.mean(losses_train), '{}-train_acc'.format(name): np.mean(accs_train) ,
    #            '{}-val_loss'.format(name)  : np.mean(losses_val), '{}-val_acc'.format(name): np.mean(accs_val),
    #            '{}-test_loss'.format(name)  : np.mean(losses_test), '{}-test_acc'.format(name): np.mean(accs_test)   })
    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}

@torch.no_grad()
def eval_gc(model, loader, mask, device):
    model.eval()


    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            loss = model.loss(pred[mask], label[mask])
            total_loss = loss.item() 
            preds.append(pred[mask])
            targets.append(label[mask])
            lss.append(total_loss)

        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0

        preds = torch.stack(preds).view(targets.size(0) , -1 )

        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc
    
    
def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_gc_prox(model, dataloader, dropout, optimizer, local_epoch, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    convGradsNorm = []
    for epoch in range(local_epoch):
        model.train()
        
        for batch in dataloader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, dropout)
            label = batch.y
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask]) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
                
                
            total_loss, acc = eval_gc(model, dataloader, batch.train_mask, device)
            loss_v, acc_v = eval_gc(model, dataloader, batch.val_mask,  device)
            loss_tt, acc_tt = eval_gc(model, dataloader, batch.test_mask, device)

            losses_train.append(total_loss)
            accs_train.append(acc)
            losses_val.append(loss_v)
            accs_val.append(acc_v)
            losses_test.append(loss_tt)
            accs_test.append(acc_tt)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test, 'convGradsNorm': convGradsNorm}

@torch.no_grad()
def eval_gc_prox(model, loader, mask, device, gconvNames, mu, Wt):
    model.eval()
    model.to(device)

    with torch.no_grad():
        targets, preds, lss = [], [], []
        for batch in loader:
            batch.to(device)
            pred = model(batch.x, batch.edge_index)
            
            label = batch.y
            loss = model.loss(pred[mask], label[mask]) + mu / 2. * _prox_term(model, gconvNames, Wt)
            total_loss = loss.item() 
            preds.append(pred[mask])
            targets.append(label[mask])
            lss.append(total_loss)
        
        targets = torch.stack(targets).view(-1)
        if targets.size(0) == 0: return  np.mean(lss) , 1.0
        preds = torch.stack(preds).view(targets.size(0) , -1 )
        preds = preds.max(1)[1]
        acc = 100 * preds.eq(targets).sum().item() / targets.size(0)
        return np.mean(lss) , acc

    