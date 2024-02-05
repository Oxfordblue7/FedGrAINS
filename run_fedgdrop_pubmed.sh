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
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        echo "Local GFN"
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --batch_size $BATCH --model GCNv2 --algo fedavg --local_flow --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
else
    if [ $LOCGFN  -eq 0 ]; then
        echo "Fed GFN"
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --overlap --batch_size $BATCH --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --overlap --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --overlap --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
    else
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --overlap --local_flow --batch_size $BATCH --model GCNv2 --algo fedavg  --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --overlap --local_flow --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
        OMP_NUM_THREADS=10  CUDA_VISIBLE_DEVICES=6 python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --overlap --local_flow --batch_size $BATCH --model GCNv2 --algo fedavg --exp_num $EXP --num_clients $NCLI -c $CFILE
    fi
fi

