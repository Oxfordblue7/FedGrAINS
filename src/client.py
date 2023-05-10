

class Client():
    def __init__(self, cid, model, data, optimizer, args):
        self.cid = cid
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.args = args

        self.Ws = {k: v for k, v in self.model.named_parameters()}

    def download_from_server(self, server):
        for k in server.Ws:
            self.Ws[k].data = server.Ws[k].data.clone()

    def local_train(self, local_epoch):
        pass

    def evaluate(self, data):
        pass
