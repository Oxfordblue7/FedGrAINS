NCLI=$1
EXPNUM=$6
DSET=$2
MODE=$3
PART=$5



#FedAvg + NA
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --exp_num $EXPNUM
