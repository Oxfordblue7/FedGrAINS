NCLI=$1
EXP=$2
BATCH=$3
CFILE='configs/cora.txt'
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --num_clients $NCLI  --exp_num $EXP -c $CFILE
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --num_clients $NCLI --exp_num $EXP -c $CFILE
