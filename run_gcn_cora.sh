NCLI=$1
DM=$2
python exps/main_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --model GCN --algo fedprox --num_clients $NCLI --dropping_method $DM
python exps/main_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --model GCN --algo fedprox --num_clients $NCLI --dropping_method $DM
python exps/main_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --model GCN --algo fedprox --num_clients $NCLI --dropping_method $DM
python exps/main_oneDS.py --repeat 4 --dataset 'Cora' --seed 4 --model GCN  --algo fedprox --num_clients $NCLI --dropping_method $DM
python exps/main_oneDS.py --repeat 5 --dataset 'Cora' --seed 5 --model GCN  --algo fedprox --num_clients $NCLI --dropping_method $DM


