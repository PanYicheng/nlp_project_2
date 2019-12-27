# virtual env activate
source scripts/activate.sh
# make vocab first
# python make_dic.py ./data/new_rocs_data_v3/raw_data/all.txt ./data/new_roc_data_v3/all-vocab.pkl
# train model
export CUDA_VISIBLE_DEVICES=2
python train.py --cuda \
	--data "./data/new_roc_data_v3/disc_data/" \
	--dic "./data/new_roc_data_v3/all-vocab.pkl" \
	--save "./model/new_roc_data_v3/aslm/aslm_e120.pt" \
	--old "./model/new_roc_data_v3/aslm/aslm.pt" \
	--emsize 1024 --nhid 1024 --nlayers 3 --tied \
	--cutoffs 4000 20000 \
	--epochs 60 --batch_size 256 --bptt 70\
	--clip 0.25 --lr 0.1
	
