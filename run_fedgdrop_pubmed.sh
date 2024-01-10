NCLI=$1
CFILE='configs/pubmed.txt'
python exps/main_fedgdrop_oneDS.py --repeat 1 --dataset 'PubMed' --seed 1 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 2 --dataset 'PubMed' --seed 2 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 3 --dataset 'PubMed' --seed 3 --model GCNv2 --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 4 --dataset 'PubMed' --seed 4 --model GCNv2  --algo fedavg --num_clients $NCLI -c $CFILE
python exps/main_fedgdrop_oneDS.py --repeat 5 --dataset 'PubMed' --seed 5 --model GCNv2  --algo fedavg --num_clients $NCLI -c $CFILE

