NCLI=$1
EXP=$2
CFILE='configs/cora.txt'
python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --model GCNv2 --algo fedavg --num_clients $NCLI  --exp_num $EXP -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --model GCNv2 --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --model GCNv2 --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 4 --dataset 'Cora' --seed 4 --model GCNv2  --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 5 --dataset 'Cora' --seed 5 --model GCNv2  --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE

