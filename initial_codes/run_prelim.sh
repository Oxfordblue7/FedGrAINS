NCLI=$1
DSET=$2
MODEL=$3
EXPNUM=$4

#metis
# cd metis-5.1.0
# make config shared=1 prefix=~/.local/
# make install
# export METIS_DLL=~/.local/lib/libmetis.so
# cd -


#FedAvg + GNN
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
#FedProx + GNN

python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM

#FedAvg +  DropEdge
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM

#FedProx +  DropEdge
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
