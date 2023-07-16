import os, sys
import argparse
import random
import torch
from pathlib import Path
import copy

from pathlib import Path

lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setupGC
from training import *

def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_{args.dropping_method}_{args.dropout}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_{args.dropping_method}_{args.dropout}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_{args.dropping_method}_{args.dropout}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_{args.dropping_method}_{args.dropout}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='FedProx  regularization parameter.')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dm', '--dropping_method', type=str, default='DropEdge',
                    help='The chosen dropping method [Dropout, DropEdge, DropNode, DropMessage,NA].')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=42)
    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs/raw',
                        help='The base path for outputting.')
    parser.add_argument('--exp_num', type=int, default=0,
                        help='Experiment number.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--dataset', help='specify the dataset',
                        type=str, default='Cora')
    parser.add_argument('--model', help='specify the model',
                        type=str, default='GCN')
    parser.add_argument('--algo', help='specify the Federated optimization',
                        type=str, default='fedprox')
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)

    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    outpath = os.path.join(args.outbase, f'{args.dataset}-{args.num_clients}clients/exp_{args.exp_num}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")


    """ distributed one dataset to multiple clients """
    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

    splitedData,num_classes = setupGC.prepareData_oneDS(args.datapath, args.dataset, num_client=args.num_clients, batchSize=args.batch_size,
                                                      convert_x=args.convert_x, seed=seed_dataSplit, overlap=args.overlap)
    print("Done")

    # save statistics of data on clients
    # if args.repeat is None:
    #     outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    # else:
    #     outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    # df_stats.to_csv(outf)
    # print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData,num_classes, args)
    print("\nDone setting up devices.")

    if args.algo == 'selftrain':
        process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=args.local_epoch)
    elif args.algo == 'fedavg':
        process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    else:
        process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=args.mu)
