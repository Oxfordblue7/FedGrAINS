
python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedprox'\
                --dropping_method 'DropEdge' \
                --dataset $4 \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed $5


python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedprox'\
                --dropping_method NA \
                --dataset $4  \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients $3\
                --seed $5
