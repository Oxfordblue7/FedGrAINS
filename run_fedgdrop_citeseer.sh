NCLI=$1
CFILE='configs/citeseer.txt'
python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 4 --dataset 'CiteSeer' --seed 4 --model GCNv2  --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 5 --dataset 'CiteSeer' --seed 5 --model GCNv2  --algo fedavg --num_clients $NCLI -c $CFILE

