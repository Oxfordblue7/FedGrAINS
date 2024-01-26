NCLI=$1
EXP=$2
BATCH=$3
LAP=$4
CFILE=$5
OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --overlap $LAP --batch_size $BATCH --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --overlap $LAP --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --overlap $LAP --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE


