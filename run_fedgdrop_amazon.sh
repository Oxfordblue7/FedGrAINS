NCLI=$1
EXP=$2
BATCH=$3
DSET=$4
LAP=$5
LOCGFN=$6
ALGO=$7
CFILE=$8

if [ $LAP -eq 0 ]; then
    echo "Disjoint"
    if [ $LOCGFN -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset $DSET --batch_size $BATCH --seed 1 --model GCNv2 --algo $ALGO --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset $DSET --batch_size $BATCH --seed 2 --model GCNv2 --algo $ALGO  --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset $DSET --batch_size $BATCH --seed 3 --model GCNv2 --algo $ALGO  --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset $DSET --batch_size $BATCH --seed 1 --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset $DSET --batch_size $BATCH --seed 2 --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset $DSET --batch_size $BATCH --seed 3 --model GCNv2 --algo $ALGO  --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
else
    echo "Overlapping"
    if [ $LOCGFN -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset $DSET --batch_size $BATCH --overlap --seed 1 --model GCNv2 --algo $ALGO --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset $DSET --batch_size $BATCH --overlap --seed 2 --model GCNv2 --algo $ALGO  --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset $DSET --batch_size $BATCH --overlap --seed 3 --model GCNv2 --algo $ALGO  --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        OMP_NUM_THREADS=5  CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset $DSET --batch_size $BATCH --overlap --seed 1 --model GCNv2 --algo $ALGO --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset $DSET --batch_size $BATCH --overlap --seed 2 --model GCNv2 --algo $ALGO  --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=5 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset $DSET --batch_size $BATCH --overlap --seed 3 --model GCNv2 --algo $ALGO  --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
fi
