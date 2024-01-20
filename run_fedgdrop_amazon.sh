NCLI=$1
EXP=$2
BATCH=$3
DSET=$4
CFILE=$5

OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset $DSET --batch_size $BATCH --seed 1 --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=50 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset $DSET --batch_size $BATCH --seed 2 --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=50 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset $DSET --batch_size $BATCH --seed 3 --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE

