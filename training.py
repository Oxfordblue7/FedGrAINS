import pandas as pd
import numpy as np
import wandb


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


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
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


