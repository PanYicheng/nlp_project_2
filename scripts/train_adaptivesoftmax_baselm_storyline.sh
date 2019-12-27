# virtual env activate
source scripts/activate.sh
# make vocab first
python make_dic.py ./data/rocstoryline_data/raw_data/all.txt ./data/rocstoryline_data/all-vocab.pkl
# train model
# export CUDA_VISIBLE_DEVICES=2
python train.py --cuda --gpu 2 \
	--data "./data/rocstoryline_data/disc_data/" \
	--dic "./data/rocstoryline_data/all-vocab.pkl" \
	--save "./model/rocstoryline_data/aslm_new_model_v1.pt" \
	--old "./model/rocstoryline_data/aslm_new_model_v1.pt" \
	--emsize 500 --nhid 1000 --nlayers 2 \
	--cutoffs 4000 20000 \
	--epochs 30 --batch_size 256 --bptt 80\
	--clip 0.25
	
