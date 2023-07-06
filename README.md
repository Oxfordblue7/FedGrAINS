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
bash run_gcn_cora.sh
```

* After running the above command lines, the raw results are stored in ```./outputs/raw/```.

* Then, to process the raw results:
```
python exps/aggregateResults.py --dataset 'Cora' --numcli 10 --mu 0.01
```

* Finally, the results are stored in ```./outputs/processed/```.

## Options
The default values for  parameters parsed to the experiment are given in ```./exps/main_oneDS.py```. Details about some of those parameters are given here.
* ```--dataset:```  The subgraph FL dataset. Default: 'Cora'. Options: 'Citeseer', 'PubMed', 'MS', .
* ```--num_rounds:``` The number of rounds for simulation. Default: 200.
* ```--local_epoch:``` The number of local epochs. Default: 1.
* ```--hidden:``` The number of hidden units. Default: 64.
* ```--nlayer:``` The number of GNN layers. Default: 3.
