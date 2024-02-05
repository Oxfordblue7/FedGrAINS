NCLI=$1
EXP=$2
BATCH=$3
LAP=$4
LOCGFN=$5
CFILE=$6

if [ $LAP -eq 0 ]; then
    echo "Disjoint"
    if [ $LOCGFN  -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --batch_size $BATCH  --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --batch_size $BATCH  --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
else
    echo "Overlapping"
    if [ $LOCGFN  -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --overlap --num_clients $NCLI  --exp_num $EXP -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --overlap --num_clients $NCLI  --exp_num $EXP -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --overlap --num_clients $NCLI --exp_num $EXP -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'CiteSeer' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'CiteSeer' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'CiteSeer' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
fi