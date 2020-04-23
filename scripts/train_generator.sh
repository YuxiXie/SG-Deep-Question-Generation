 #!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src
MODELHOME=${HOME}/models
LOGHOME=${HOME}/logs

mkdir -p ${MODELHOME}
mkdir -p ${LOGHOME}

cd ${EXEHOME}

python train.py \
       -sequence_data ${DATAHOME}/preprcessed-data/preprcessed_sequence_data.pt \
       -graph_data ${DATAHOME}/preprcessed-data/preprcessed_graph_data.pt \
       -train_dataset ${DATAHOME}/Datasets/train_dataset.pt \
       -valid_dataset ${DATAHOME}/Datasets/valid_dataset.pt \
       -checkpoint ${MODELHOME}/classifier_84.06773_accuracy.chkpt \
       -epoch 100 \
       -batch_size 32 -eval_batch_size 16 \
       -pre_trained_vocab \
       -training_mode generate \
       -max_token_src_len 200 -max_token_tgt_len 50 \
       -sparse 0 \
       -copy \
       -coverage -coverage_weight 0.4 \
       -node_feature \
       -d_word_vec 300 \
       -d_seq_enc_model 512 -d_graph_enc_model 256 -n_graph_enc_layer 3 \
       -d_k 64 -brnn -enc_rnn gru \
       -d_dec_model 512 -n_dec_layer 1 -dec_rnn gru \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.5 -attn_dropout 0.1 \
       -gpus 0 \
       -save_mode best -save_model ${MODELHOME}/generator \
       -log_home ${LOGHOME} \
       -logfile_train ${LOGHOME}/train_generator \
       -logfile_dev ${LOGHOME}/valid_generator \
       -translate_ppl 15 \
       -curriculum 0  -extra_shuffle -optim adam \
       -learning_rate 0.00025 -learning_rate_decay 0.75 \
       -valid_steps 500 \
       -decay_steps 500 -start_decay_steps 5000 -decay_bad_cnt 5 \
       -max_grad_norm 5 -max_weight_value 32
