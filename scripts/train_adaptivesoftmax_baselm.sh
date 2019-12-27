# virtual env activate
source scripts/activate.sh
# make vocabulary
python make_dic.py ./data/rocstory_data/all.txt ./data/rocstory_data/all-vocab.pkl
# train model
python train.py --cuda --gpu 0 \
	--data "./data/rocstory_data/" \
	--dic "./data/rocstory_data/all-vocab.pkl" \
	--save "./model/rocstory_data/aslm_model.pt" \
	--emsize 500 --nhid 1000 --nlayers 2 \
	--cutoffs 4000 20000 \
	--epochs 30 --batch_size 64 --bptt 50
	
