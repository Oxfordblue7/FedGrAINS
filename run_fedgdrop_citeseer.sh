NCLI=$1
EXP=$2
BATCH=$3
CFILE='configs/citeseer.txt'
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=50  CUDA_VISIBLE_DEVICES=3 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
