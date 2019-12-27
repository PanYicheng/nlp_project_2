#!/bin/bash
# virtual env activate
source scripts/activate.sh
export CUDA_VISIBLE_DEVICES=2
# make vocab first
python make_dic.py ./data/new_roc_data/raw_data/all.txt ./data/new_roc_data/all-vocab.pkl

python project_1_main.py --data ./data/new_roc_data/disc_data/ \
	--dic ./data/new_roc_data/all-vocab.pkl \
	--save ./model/new_roc_data/project1_model_new_roc_data_e100.pt \
	--resume ./model/new_roc_data/project1_model_new_roc_data.pt \
	--model GRU --emsize 500 --nhid 500 --nlayers 3 --tied \
	--lr 10 --clip 0.25 --epochs 50 --batch_size 100 --bptt 70 \
	--alpha 0.1 --wdrop 0 \
	--dropoute 0.5 --dropouti 0.5 --dropout 0.5 --dropoutrnn 0.5 \
	--seed 42 --nonmono 5 --cuda --log-interval 200
