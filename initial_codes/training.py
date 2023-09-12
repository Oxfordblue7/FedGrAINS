import pandas as pd
import numpy as np


def run_selftrain_NC(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        client.stats['testLosses'].append(loss)
        client.stats['testAccs'].append(acc)
        allAccs[client.name] = [max(client.train_stats['trainingAccs']), max(client.train_stats['valAccs']), max(client.train_stats['testAccs'])]
        print("  > {} done.".format(client.name))

    return allAccs


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            np.random.seed(c_round)
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)
            testLoss, testAcc = client.evaluate()
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testAcc)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

    frame = pd.DataFrame()
    for client in clients:
        frame.loc[client.name, 'train_acc'] =  max(client.stats['trainingAccs'])
        frame.loc[client.name, 'val_acc'] =  max(client.stats['valAccs'])
        m =  max(client.stats['valAccs'])
        frame.loc[client.name, 'test_acc'] = client.stats['testAccs'][client.stats['valAccs'].index(m)]


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

        for client in selected_clients:
            client.local_train_prox(local_epoch, mu)
            testLoss, testAcc = client.evaluate_prox()
            client.stats['testLosses'].append(testLoss)
            client.stats['testAccs'].append(testAcc)

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

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


