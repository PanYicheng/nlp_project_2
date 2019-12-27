#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python project_1_main.py --data ./data/rocstoryline_data/disc_data/ \
	--dic ./data/rocstoryline_data/all-vocab.pkl \
	--save ./model/rocstoryline_data/project1_model.pt \
	--resume ./model/rocstoryline_data/project1_model.pt \
	--model GRU --emsize 500 --nhid 500 --nlayers 3 --tied \
	--lr 0 --clip 0.25 --epochs 1 --batch_size 100 --bptt 70 \
	--alpha 0.1 --wdrop 0 \
	--dropoute 0.5 --dropouti 0.5 --dropout 0.5 --dropoutrnn 0.5 \
	--seed 42 --nonmono 5 --cuda --log-interval 200
