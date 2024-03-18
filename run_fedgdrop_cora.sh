NCLI=$1
EXP=$2
BATCH=$3
LAP=$4
LOCGFN=$5
ALGO=$6
CFILE=$7

if [ $LAP -eq 0 ]; then
    echo "Disjoint"
    if [ $LOCGFN -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --batch_size $BATCH  --model GCNv2 --algo $ALGO --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --batch_size $BATCH --model GCNv2 --algo $ALGO --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=4 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --batch_size $BATCH  --model GCNv2 --algo $ALGO --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
else
    echo "Overlapping"
    if [ $LOCGFN -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --batch_size $BATCH --model GCNv2 --algo $ALGO --overlap --num_clients $NCLI  --exp_num $EXP -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --batch_size $BATCH --model GCNv2 --algo $ALGO --overlap --num_clients $NCLI  --exp_num $EXP -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=3  python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --batch_size $BATCH --model GCNv2 --algo $ALGO --overlap --num_clients $NCLI --exp_num $EXP -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'Cora' --seed 1 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'Cora' --seed 2 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'Cora' --seed 3 --batch_size $BATCH --model GCNv2 --algo $ALGO --local_flow --overlap --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
fi