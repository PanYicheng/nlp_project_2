# Steps

Steps logged here to finish the whole project2.

## Convert data

Using ./data_process.ipynb

1. Convert the rocstory to the needed data form:

   * ```title <EOT> </s> story```
   * ```title <EOT> </s> l1 ... l5 <EOL> </s> s1 </s> ... s5 </s>```

2. Also generate a **all.txt** containing all data for the dictionary.

3. Count the average story length for the bptt parameter. Here for the storyline data max and min story length are 123 and 38.

4. Calculate the statistics of word frequency to set a appropriate cutoff for adaptive softmax. Here the cutoff is 4000, 20000.

Data is ouput to directory ```./data/```**${data_spec}**```/raw_data/``` and ```./data/```**${data_spec}**```/raw_data/```.

## Using ./preprocess_data/split_disc_data.py

Split the train data to train and discriminator data.

Run ```python ./preprocess_data/split_disc_data.py ./data/```**${data_spec}**```/raw_data/train.txt ./data/```**${data_spec}**```/disc_data/``` to split input train txt file to train and disc txt. The default fraction is 0.8, 0.2.

Also copy the valid.txt and test.txt from ```raw_data``` to the output directory ```./data/```**${data_spec}**```/disc_data/```.

## Train base language model

Using one of:

* ```bash scripts/train_adaptivesoftmax_baselm_storyline.sh``` 
* ```bash scripts/train_project1_lm.sh```
* ```bash scripts/train_project1_lm_new_roc_data.sh```
* ```bash scripts/train_p1lm_new_roc_v2.sh```
* ```bash scripts/train_p1lm_new_roc_v3.sh```

to train a  
language model with the ```title storyline story``` data.

The shell scripts will source the environment and generate the vocab to ```./data/```**${data_spec}**```/all-vocab.pkl```. Vocab will be used later again.

Model                          | Detail                                |Train loss | Valid loss | Test loss
|-|-|-|-|-|
aslm_new_model_v1.pt           | Default                               | 3.9065    |  3.9721    | 3.91
aslm_new_model_v2.pt           | use ASGD optimizer                    |           |            |
aslm_new_model_v2_continued.pt | use ASGD optimizer and LR scheduler   | 4.4071    |  4.4512    | 4.41
aslm_model_v3.pt               | Default (scripts/train_aslm_v3.sh)    | 3.7639    |  3.8426    | 3.78
aslm.pt  | (scripts/train_aslm_new_roc_v3.sh)    |4.1085  | 4.1724   | 4.13
aslm_e120.pt  | (scripts/train_aslm_new_roc_v3.sh)    |4.02|4.11|4.06
|project1_model_new_roc_data.pt | P1 model on new roc data | 4.61 | 3.76 | 3.68 |
|project1_model_new_roc_data_e100.pt | P1 model on new roc data | 4.50 | 3.65 | 3.58
|p1lm_new_roc_data_v3.pt | P1 model on new roc data v3 | 4.38 | 3.60 | 3.53 |



## Make context and continuation dataset

Run ```python make_cc_version.py ./data/```**${data_spec}**```/disc_data/ --len_context 2 --len_continuation 5 --doc_level```
to generate contexts and continuations. 

This step will generate ```*.context, *.true_continuation, *.shuffled_continuation```file for each dataset.

> Note: The rocstoryline data is in format ```title <EOT> </s> l1 ... l5 <EOL> </s> s1 </s> ... s5 </s>```. So we use the title and storyline as context and the remaining 5 sentences as continuation. So we use a split of 2, 5.

## Use lm to generate discriminator data

Run ```bash scripts/gen_lm_data.sh ./data/```**${data_spec}**```/disc_data/ ./model/```**${data_spec}**```/```**${run_spec}**```.pt ./data/```**${data_spec}**```/all-vocab.pkl``` to generate discriminator data from the language model.

The input is ```*.context```. The output file is ```*.generated_continuation```.

The out is located in the same disc directory. So it's better to put generated txts to different subfolders to separate them between different runs.

> All the following procedures should use the same **${run_spec}**, current values are:

|Run Spec|Note|
|-|-|
|aslm_new_model_v1| The original model
|project1_model   | The model from nlp project 1
|project1_model_new_roc_data| Project 1 model with new data
|p1lm_new_roc_data_v3 | P1 model on new roc data v3
|aslm_e120 | aslm on new roc data v3

## Make repetition data

Run this to generate repetition data:
```python scripts/create_classifier_dataset.py ./data/```**${data_spec}**```/disc_data/```**${run_spec}** ```./data/```**${data_spec}**```/rep_data/```**${run_spec}**/ ```--comp lm```.

The input txt in the argument directory is assumed to be ```*.context,*.generated_continuation,*.true_continuation```.
The out data is named ```*.tsv``` in the out directory.

```*``` is in ```disc_train, valid, test```

## Train repetition model

Run ```python train_classifier.py ./data/```**${data_spec}**```/rep_data/```**${run_spec}/** ```--save_to ./model/```**${data_spec}**/```repetition_model_```**${run_spec}**```.pt --dic ./data/```**${data_spec}**```/all-vocab.pkl  --fix_embeddings --adam --ranking_loss --train_prefixes --decider_type reprnn```

Train results for model:
|Rep Model|Valid Acc|Train Acc
|-|-|-|
|v1|1.0|1.0
|project1_model|0.93|1.00
|project1_model_new_roc_data |0.97 |0.99
|p1lm_new_roc_data_v3 | 1.00|0.98
|p1lm_new_roc_data_v3_norank_loss | 0.95|0.98
|new_roc_data_v3/aslm/rep_model.pt |1.0|0.84


## Skip entailment model

This model is thoughted to be of no use in this task.
Maybe consider merge the pretrained model.

## Make relevance data

Run ```python scripts/create_classifier_dataset.py ./data/```**${data_spec}**```/disc_data/```**${run_spec}**/ ```./data/```**${data_spec}**```/rel_data/```**${run_spec}**/ ```--comp random```

## Train relevance model

Run ```python train_classifier.py ./data/```**${data_spec}**```/rel_data/```**${run_spec}**/ ```--save_to ./model/```**${data_spec}**```/relevance_model_```**${run_spec}**```.pt --dic ./data/```**${data_spec}**```/all-vocab.pkl --decider_type cnncontext --adam  --ranking_loss --train_prefixes --num_epochs 20```

train result:

Train results for model:
|Rel Model|Valid Acc|Train Acc
|-|-|-|
|v1|0.615400|0.921063
|project1_model|0.71|0.97
|project1_model_new_roc_data |1.00 |0.81
|p1lm_new_roc_data_v3 | 1.00|0.83
|p1lm_new_roc_data_v3_norankloss | 0.98|0.65
|new_roc_data_v3/aslm/rel_model.pt |0.98|0.99

## Lexical style model

The lexical style uses the same data as in the repetition model.

Run ```python train_classifier.py ./data/```**${data_spec}**```/rep_data/```**${run_spec}** ```--save_to ./model/```**${data_spec}**```/lexical_model_```**${run_spec}**```.pt --dic ./data/```**${data_spec}**```/all-vocab.pkl --decider_type poolending --adam --ranking_loss --train_prefixes```

Train results for model:
|Lexical Model|Valid Acc|Train Acc
|-|-|-|
|v1|1.00|0.98
|project1_model |0.99 |0.99
|project1_model_new_roc_data |1.00 |0.96
|p1lm_new_roc_data_v3 | 1.00|0.98
|p1lm_new_roc_data_v3_norankloss | 0.98|0.99
|new_roc_data_v3/aslm/rel_model.pt |0.99|1.00

## Train discriminitor weights

Make a weight file, with the following format(tab separated)
> 1 SCORER_PATH	SCORER_CLASS	/path/to/model.pt

SCORER_PATH and SCORER_CLASS are word_rep.context_scorer & ContextScorer respectively for all modules, except the entailment module.

For the entailment module SCORER_PATH and SCORER_CLASS are entailment.entail_scorer_new & EntailmentScorer. Because we didnot use it,
just ignore it.

Now run this to generate the weight train data.
```python scripts/create_classifier_dataset.py ./data/```**${data_spec}**```/disc_data/```**${run_spec}**/ ```./data/```**${data_spec}**```/weight_data/```**${run_spec}**/ ```--comp none```

And then train it:
```python generate.py --cuda --data ./data/```**${data_spec}**```/weight_data/```**${run_spec}**```/valid.tsv --lm ./model/```**${data_spec}**```/```**${run_spec}**```/lm_model.pt --dic ./data/```**${data_spec}**```/all-vocab.pkl  --scorers ./model/```**${data_spec}**/**${run_spec}**```/scorer_weight.tsv --print --learn --ranking_loss --save_every 10 --lr 1```

Finally generate on the test data:
```python generate.py --cuda --data ./data/```**${data_spec}**```/weight_data/```**${run_spec}**```/test.tsv --out ./data/```**${data_spec}**```/weight_data/```**${run_spec}**```/test.tsv.out --lm ./model/```**${data_spec}/${run_spec}**```/lm_model.pt --dic ./data/```**${data_spec}**```/all-vocab.pkl  --scorers ./model/```**${data_spec}/${run_spec}**```/scorer_weight.tsv --print```

## Final bleu score

| Data | Bleu |
| -    | -    |
|./data/rocstoryline_data/weight_data/project1_model/test.tsv.out | 0.030897132412493093
| ./data/new_roc_data/weight_data/project1_model_new_roc_data/test.tsv.out | 0.009508469036361263