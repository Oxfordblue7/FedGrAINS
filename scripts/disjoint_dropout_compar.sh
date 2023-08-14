python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method 'DropEdge' \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 42


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method 'DropEdge' \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 256


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method 'DropEdge' \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 431


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method NA \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 42


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method NA \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 256


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dropping_method NA \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed 431
