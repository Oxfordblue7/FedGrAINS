import random
import torch

class Server():
    def __init__(self, model):
        self.model = model

        self.Ws = {k: v for k, v in self.model.named_parameters()}

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_avg(self, clients):
        total_size = 0
        for client in clients:
            total_size = total_size + client.data['train_g_pos'].number_of_edges() + client.data[
                'train_g_neg'].number_of_edges()
        for k in self.Ws:
            self.Ws[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.Ws[k].data,
                                                                         client.data['train_g_pos'].number_of_edges() +
                                                                         client.data['train_g_neg'].number_of_edges())
                                                               for client in clients]), dim=0), total_size)
