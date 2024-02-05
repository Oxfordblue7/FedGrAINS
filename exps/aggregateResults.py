import os
import wandb
import argparse
import pandas as pd
from pathlib import Path

def _aggregate(inpath, outpath, filename):
    dfs = []
    for file in os.listdir(inpath):
        print(file)
        if file.endswith(filename):
            dfs.append(pd.read_csv(os.path.join(inpath, file), header=0, index_col=0))
    df = pd.concat(dfs)
    group = df.groupby(df.index)
    dfmean = group.mean()
    dfstd = group.std()
    df_out = dfmean.join(dfstd, lsuffix='_mean', rsuffix='_std')
    df_out.to_csv(os.path.join(outpath, filename), header=True, index=True)


def average_aggregate_all(args):
    #, f'fedprox_mu{args.mu}_NA', f'fedprox_mu{args.mu}_Dropout', f'fedprox_mu{args.mu}_DropEdge'
    #'selftrain_NA','selftrain_Dropout' ,'selftrain_DropEdge', 'fedavg_NA','fedavg_Dropout' ,
    # 'avg-glob-test_acc_mean', 'avg-glob-test_acc_std' , 'globtest_acc_mean', 'globtest_acc_std'
    algos = ['fedavg']
    dfs = pd.DataFrame(index=algos, columns=['avg-val_acc_mean', 'avg-val_acc_std', 'avg-test_acc_mean', 'avg-test_acc_std'])
    for algo in algos:
        df = pd.read_csv(os.path.join(args.outpath, f'accuracy_{algo}_64_GC.csv'), header=0, index_col=0)
        df = df[['val_acc_mean', 'val_acc_std', 'test_acc_mean', 'test_acc_std']]
        print(df.shape)
        dfs.loc[algo] = list(df.mean())
    wandb_agg_df = wandb.Table(dataframe=dfs)
    wandb.log({"aggr_results" : wandb_agg_df})
    outfile = os.path.join(args.outpath, f'avg_accuracy_allAlgos_GC.csv')
    dfs.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)

def main_aggregate_all_multiDS(args):
    """ multiDS: aggregagte all outputs """
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    for filename in [ 'accuracy_fedavg_GC.csv' , 'accuracy_fedprox_mu'+str(args.mu)+'_GC.csv']:
        _aggregate(args.inpath, args.outpath, filename)

    """ get average performance for all algorithms """
    average_aggregate_all(args)


def main_aggregate_prelim(args):
    """ multiDS: aggregagte all outputs """
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    # f'accuracy_selftrain_NA_{args.dropout}_GC.csv', f'accuracy_selftrain_Dropout_{args.dropout}_GC.csv', f'accuracy_selftrain_DropEdge_{args.dropout}_GC.csv',
    # f'accuracy_fedavg_NA_{args.dropout}_GC.csv' , f'accuracy_fedavg_Dropout_{args.dropout}_GC.csv' ,
    #  f'accuracy_fedprox_mu{args.mu}_NA_{args.dropout}_GC.csv', f'accuracy_fedprox_mu{args.mu}_Dropout_{args.dropout}_GC.csv', f'accuracy_fedprox_mu{args.mu}_DropEdge_{args.dropout}_GC.csv'

    for filename in [ f'accuracy_fedavg_64_GC.csv']:
        _aggregate(args.inpath, args.outpath, filename)

    """ get average performance for all algorithms """
    average_aggregate_all(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='specify the dataset',
                        type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'CS', 'Physics'])
    parser.add_argument('--numcli', help='specify the number of clients',
                        type=int, default=10, choices=[3,5,10,20,30,50])
    parser.add_argument('--mu', help='specify the FedProx parameter',
                        type=float, default=0.001)
    parser.add_argument('--exp_num', type=int, default=0,
                        help='Experiment number.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--model', help='specify the model',
                        type=str, default='GCNv2')
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    args.inpath = f'./outputs/raw/{args.dataset}-METIS-overlap-{args.numcli}clients-{args.model}-wlocal_gfn/exp_{args.exp_num}/repeats'
    args.outpath = f'./outputs/processed/{args.dataset}-METIS-overlap-{args.numcli}clients-{args.model}-wlocal_gfn/exp_{args.exp_num}'
    wandb.init( project = 'fedgdrop',
        name=f'{args.dataset}-{args.numcli}clients-{args.model}-aggr-results-{args.exp_num}' 
    )
    wandb.config.update(args)
    #     """ multiDS: aggregagte all outputs """
    main_aggregate_prelim(args)
    wandb.finish()
