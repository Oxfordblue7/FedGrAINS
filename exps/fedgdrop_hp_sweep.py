import os, sys, json
import argparse, configparser
import random
import torch
import wandb
import math
from pathlib import Path
import copy
import numpy
from pathlib import Path

lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setup_devices
from training import *
# wandb.login(relogin = True)

def process_fedavg(clients, server, args = None):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    outfile = os.path.join(args.outpath, f'accuracy_fedavg_{args.batch_size}_GC{args.suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcfl(clients, server, args = None):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    outfile = os.path.join(args.outpath, f'accuracy_gcfl_{args.batch_size}_GC{args.suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcfl_nc(clients, server, args = None):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL(CLF only) ...")
    
    outfile = os.path.join(args.outpath, f'accuracy_gcfl_nc_{args.batch_size}_GC{args.suffix}.csv')

    frame = run_gcfl_nc(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def process_gcflplus(clients, server, args = None):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")
    outfile = os.path.join(args.outpath, f'accuracy_gcflplus_{args.batch_size}_GC{args.suffix}.csv')
    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1 = 0.05, EPS_2 = 0.1, seq_length = args.seq_length, standardize = False)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--algo', help='specify the Federated optimization',
                        type=str, default='fedavg')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of GNN layers') 
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for inner solver;')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=42)
    parser.add_argument('--natural_split', help='use public split to partition',
                        type=bool, default=False)
    parser.add_argument('--datapath', type=str, default='../datasets',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs/raw',
                        help='The base path for outputting.')
    parser.add_argument('--exp_num', type=int, default=100,
                        help='Experiment number.')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--partition', help='Graph partitioning algorithm',
                        type=str, default="METIS")
    parser.add_argument('--dataset', help='specify the dataset',
                        type=str, default='Cora')
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)

    
    """ 
    GFlowNet Parameters
    """
    parser.add_argument('--n_hops', type=int, default=2,
                        help='The number of hops to sample around.')
    parser.add_argument('--use_indicators', type=bool, default=True,
                        help='The number of nodes to choose.')
    parser.add_argument('--eval_on_cpu', type=bool, default=True,
                        help='The number of nodes to choose.')
    parser.add_argument('--eval_full_batch', type=bool, default=True,
                        help='Option to evaluate over full batch.')
    parser.add_argument('--log_z_init', type=float, default=0.,
                        help='The logz constant initialization.')
    parser.add_argument('--reg_param', type=float, default=0.,
                        help='Regularization coefficient over')
    parser.add_argument('--local_flow', help='GFN is not federated',
                        type=bool, default=False)
    
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed_dataSplit = 1234
    args = vars(args)
    return args


def main_sweep(config= None):
   
    wandb.init( project = 'gflownet-tuning' ,
               entity = 'emirceyani')
    args  = wandb.config

    ov = "overlap" if args.overlap else "disjoint"
    gfn = "local_gfn" if args.local_flow else "fed_gfn"
    args.outpath = os.path.join(args.outbase, f'{args.dataset}-{args.algo}-{args.partition}-{ov}-{args.num_clients}clients-{args.model}-w{gfn}/exp_{args.exp_num}')
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {args.outpath}")

    """ distributed one dataset to multiple clients """
    if not args.convert_x:
        """ using original features """
        args.suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        args.suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")


    splitedData, num_features, num_classes = setup_devices.prepareData_fedgdrop_oneDS(args.datapath, args.dataset, num_client=args.num_clients, batchSize=args.batch_size,
                                                       mode = 'v2', partition = args.partition,  seed=1234, overlap=args.overlap)
    print("Done")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_clients, init_server, init_idx_clients = setup_devices.setup_fedgdrop_devices(splitedData, num_features, num_classes, args)
    print("\nDone setting up devices.") 

    if args.algo == 'fedavg':
        process_fedavg(clients=init_clients, server=init_server, args =args)
    elif args.algo == 'gcfl':
        process_gcfl(clients=init_clients, server=init_server, args =args)
    elif args.algo == 'gcfl_nc':
        process_gcfl_nc(clients=init_clients, server=init_server, args =args)
    else:
        process_gcflplus(clients=init_clients, server=init_server, args =args)
    

if __name__ == '__main__':

    args = parse_args()

    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "valAcc"},
        "parameters": {
            "batch_size": {'value': 64},
            "model": {'value': "GCNv2"},
            "flow_lr": { 'max': 0.01, 'min': 0.000001 },
            "loss_coef": {"distribution": "q_log_uniform" , "max":math.log(1e6), "min": math.log(1e2), 'q': 1},
            "k": {"distribution": "q_log_uniform", "max": math.log(64), "min": math.log(16), 'q': 1},
            "seq_length" : {'values' : [5,10]}
        }
    }
    # 'distribution': 'q_log_uniform',
    #                     'max': math.log(256),
    #                     'min': math.log(32),
    #                     'q': 1
    # { "distribution": "uniform" , "max": 0.01, "min": 0.000001}



    sweep_config['parameters'].update({key :{"value" : val} for key, val in args.items() if key not in sweep_config['parameters'].keys()})
    ov = "overlap" if args["overlap"] else "disjoint"
    gfn = "local" if args["local_flow"] else "fed"
    
    sweep_config['name'] = f'{args["algo"]}-{args["dataset"]}-{args["partition"]}-{args["num_clients"]}-{ov}-{gfn}-GFlowNet-Sweep'
    # import pprint
    # pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep = sweep_config, project="gflownet-tuning",  entity = 'emirceyani')

    # wandb.config.update(args)
    wandb.agent(sweep_id, function=main_sweep, count=100)
    #_v2
    

    wandb.finish()
