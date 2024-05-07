import os, sys, json
import argparse, configparser
import random
import torch
import wandb
from pathlib import Path
import copy

from pathlib import Path

lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setup_devices
from training import *

def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_NC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc', f'globtest_acc']] = v
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_{args.batch_size}_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_{args.batch_size}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_pfedavg(clients, server, num_features):
    print("\nDone setting up PerFedAvg devices.")

    print("Running Per-FedAvg ...")
    frame = run_pfedavg(clients, server, args.num_rounds, args.local_epoch, samp=None, n_feats = num_features)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_pfedavg_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_pfedavg_{args.batch_size}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_fedpub(clients, server, num_features):
    print("\nDone setting up PerFedAvg devices.")

    print("Running FedPUB...")
    frame = run_fedpub(clients, server, args.num_rounds, args.local_epoch, samp=None, n_feats = num_features)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedpub_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedpub_{args.batch_size}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_{args.batch_size}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_{args.batch_size}_GC{suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcfl_nc(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_{args.batch_size}_GC{suffix}.csv')

    frame = run_gcfl_nc(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_{args.batch_size}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_{args.batch_size}_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1, seq_length = args.seq_length, standardize = False)
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
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='FedProx  regularization parameter.')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=42)
    parser.add_argument('--partition', help='Graph partitioning algorithm',
                        type=str, default="METIS")
    parser.add_argument('--datapath', type=str, default='../datasets',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs/raw',
                        help='The base path for outputting.')
    parser.add_argument('--exp_num', type=int, default=4,
                        help='Experiment number.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--dataset', help='specify the dataset',
                        type=str, default='Cora')
    parser.add_argument('--model', help='specify the model',
                        type=str, default='GCN')
    parser.add_argument('--laye-mask-one', action='store_true', default=False)
    parser.add_argument('--clsf-mask-one', action='store_true', default=False)
    parser.add_argument('--algo', help='specify the Federated optimization',
                        type=str, default='fedavg')
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        action = "store_true", default=False)
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--log_wandb', help='log into wandb or not',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--l1', type=float, default=1e-3)

    
    parser.add_argument("-c", "--config_file", type=str, help='Config file')

    
    """ 
    GFlowNet Parameters
    """
    parser.add_argument('--n_hops', type=int, default=2,
                        help='The number of hops to sample around.')
    parser.add_argument('--flow_lr', type=float, default=1e-4,
                        help='learning rate for GFlowNet')
    parser.add_argument('--k', type=int, default=16,
                        help='The number of nodes to choose.')
    parser.add_argument('--use_indicators', type=bool, default=True,
                        help='The number of nodes to choose.')
    parser.add_argument('--eval_on_cpu', type=bool, default=True,
                        help='The number of nodes to choose.')
    parser.add_argument('--eval_full_batch', type=bool, default=True,
                        help='Option to evaluate over full batch.')
    parser.add_argument('--loss_coef', type=float, default=1e4,
                        help='The loss coefficient to balance reward.')
    parser.add_argument('--log_z_init', type=float, default=0.,
                        help='The logz constant initialization.')
    parser.add_argument('--reg_param', type=float, default=0.,
                        help='Regularization coefficient over')
    parser.add_argument('--local_flow', help='GFN is not federated',
                        action = "store_true", default=False)
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
# sh run_fedgdrop_cora.sh 10 1 64 0 0 gcflplus configs/gcflplus_cora_10_fedgfn_disjoint.txt
# sh run_fedgdrop_seer.sh 20 1 64 0 0 gcflplus configs/gcflplus_seer_20_fedgfn_disjoint.txt
# sh run_fedgdrop_pubmed.sh 20 1 64 0 0 gcflplus configs/gcflplus_med_20_fedgfn_disjoint.txt
        #  sh run_fedgdrop_amazon.sh 20 1 64 Computers 0 0 gcflplus configs/gcflplus_comp_20_fedgfn_disjoint.txt 
    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        defaults = {}
        defaults.update(dict(config.items("Params")))
        parser.set_defaults(**defaults)
        args = parser.parse_args() # Overwrite arguments
    seed_dataSplit = 1234

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init( project = 'fedgdrop',
        name=f'{args.dataset}-{args.num_clients}clients-{args.model}-fedgdrop',
        mode='online' if args.log_wandb else 'disabled',
        config = args
    )
    # wandb.config.update(args)
    
    #_v2
    ov = "overlap" if args.overlap else "disjoint"
    gfn = "local_gfn" if args.local_flow else "fed_gfn"
    outpath = os.path.join(args.outbase, f'{args.dataset}-{args.algo}-{args.partition}-{ov}-{args.num_clients}clients-{args.model}-w{gfn}/exp_{args.exp_num}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    with open(outpath +'/params.txt', 'w') as convert_file:
        convert_file.write(json.dumps(vars(args)))

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
    splitedData, num_features, num_classes = setup_devices.prepareData_fedgdrop_oneDS(args.datapath, args.dataset, num_client=args.num_clients, batchSize=args.batch_size,
                                                        partition = args.partition,  seed=seed_dataSplit, overlap=args.overlap)
    print("Done")

    # save statistics of data on clients
    # if args.repeat is None:
    #     outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    # else:
    #     outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    # df_stats.to_csv(outf)
    # print(f"Wrote to {outf}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_clients, init_server, init_idx_clients = setup_devices.setup_fedgdrop_devices(splitedData, num_features, num_classes, args)
    print("\nDone setting up devices.")

    if args.algo == 'selftrain':
        #They do not communicate, so they have to run num_rounds x 1 local epochs
        process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=args.num_rounds)
    elif args.algo == 'fedavg':
        process_fedavg(clients=init_clients, server=init_server)
    elif args.algo == 'fedprox':
        process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=args.mu)
    elif args.algo == 'pfedavg':
        process_pfedavg(clients=init_clients, server=init_server, num_features = num_features)
    elif args.algo == 'fedpub':
        process_fedpub(clients=init_clients, server=init_server, num_features = num_features)
    elif args.algo == 'gcfl':
        process_gcfl(clients=init_clients, server=init_server)
    elif args.algo == 'gcfl_nc':
        process_gcfl_nc(clients=init_clients, server=init_server)
    elif args.algo == 'gcflplus':
        process_gcflplus(clients=init_clients, server=init_server)
    else:
        print("Quit")
    wandb.finish()
