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
#Selftrain + NA
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM

#Selftrain +Dropout
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM

#Selftrain + DropEdge
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo selftrain --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM

#FedAvg + NA
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM

#FedAvg + Dropout
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM

#FedAvg +  DropEdge
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedavg --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM

#FedProx + NA
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method NA --exp_num $EXPNUM

#FedProx +  Dropout
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method Dropout --exp_num $EXPNUM

#FedProx +  DropEdge
python exps/main_oneDS.py --repeat 1 --dataset $DSET --seed 41 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 2 --dataset $DSET --seed 42 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
python exps/main_oneDS.py --repeat 3 --dataset $DSET --seed 43 --model $MODEL --algo fedprox --num_clients $NCLI --dropping_method DropEdge --exp_num $EXPNUM
