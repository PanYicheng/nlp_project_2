#!/bin/bash
# virtual env activate
source scripts/activate.sh
export CUDA_VISIBLE_DEVICES=1
# make vocab first
python make_dic.py ./data/new_roc_data_v2/raw_data/all.txt ./data/new_roc_data_v2/all-vocab.pkl

python project_1_main.py --data ./data/new_roc_data_v2/disc_data/ \
	--dic ./data/new_roc_data_v2/all-vocab.pkl \
	--save ./model/new_roc_data_v2/p1lm_new_roc_v2.pt \
	--model GRU --emsize 500 --nhid 500 --nlayers 3 --tied \
	--lr 10 --clip 0.25 --epochs 100 --batch_size 100 --bptt 70 \
	--alpha 0.1 --wdrop 0 \
	--dropoute 0.5 --dropouti 0.5 --dropout 0.5 --dropoutrnn 0.5 \
	--seed 42 --nonmono 5 --cuda --log-interval 200 \
	--when 90
