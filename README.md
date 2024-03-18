# FedGDrop

## Requirements & Data Preparation
To install requirements
```
conda env create -f fedgdrop.yaml
```
Then, METIS (for data generation), please follow  ```https://github.com/james77777778/metis_python```

## Running examples
* OneDS: Distributing one dataset to a number of clients:

```
sh run_fedgdrop_cora.sh 10 1 64 0 0 gcflplus configs/gcflplus_cora_10_fedgfn_disjoint.txt
```

Script repeats the experiment three times with different seeds.
You can find scripts for each datasets

### Parameters
```
num of clients=$1
experiment number (for logging results)=$2
number of batches=$3
Dataset =$4 (Only for un_fedgdrop_amazon.sh)
Enable overlap=$5
Enable local training of flow network=$6
FL Algorithm =$7
Config File=$8
```

## Running examples
* After running the above command lines, the raw results are stored in ```./outputs/raw/```.

* Then, to process the raw results:
```
python exps/aggregateResults.py --dataset 'Cora' --algo fedavg --model GCNv2 --numcli 5 --exp_num 1 
```

* Finally, the results are stored in ```./outputs/processed/```.

## Options
The default values for  parameters parsed to the experiment are given in ```./exps/main_oneDS.py```. Details about some of those parameters are given here.
* ```--dataset:```  The subgraph FL dataset. Default: 'Cora'. Options: 'Citeseer', 'PubMed', 'MS', .
* ```--num_rounds:``` The number of rounds for simulation. Default: 200.
* ```--local_epoch:``` The number of local epochs. Default: 1.
* ```--hidden:``` The number of hidden units. Default: 64.
* ```--nlayer:``` The number of GNN layers. Default: 3.


## Parameter tuning 

Under ```exps/```, ```fedgdrop_hp_sweep.py``` tunes GFlowNet parameters over random 100 runs in the given range of Python script. 
