#!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src

cd ${EXEHOME}

python preprocess.py \
       -train_src ${DATAHOME}/train/train.src.txt -train_tgt ${DATAHOME}/train/train.tgt.txt \
       -valid_src ${DATAHOME}/valid/valid.src.txt -valid_tgt ${DATAHOME}/valid/valid.tgt.txt \
       -train_ans ${DATAHOME}/train/train.ans.txt -valid_ans ${DATAHOME}/valid/valid.ans.txt \
       -train_graph ${DATAHOME}/train/train.tag.json -valid_graph ${DATAHOME}/valid/valid.tag.json \
       -node_feature \
       -copy \
       -answer \
       -save_sequence_data ${DATAHOME}/preprcessed_sequence_data.pt \
       -save_graph_data ${DATAHOME}/preprcessed_graph_data.pt \
       -train_dataset ${DATAHOME}/Dataset/train_dataset.pt \
       -valid_dataset ${DATAHOME}/Dataset/valid_dataset.pt \
       -src_seq_length 200 -tgt_seq_length 50 \
       -src_vocab_size 50000 -tgt_vocab_size 50000 \
       -src_words_min_frequency 3 -tgt_words_min_frequency 2 \
       -vocab_trunc_mode frequency \
       -pre_trained_vocab ${GLOVEHOME}/glove.840B.300d.txt -word_vec_size 300 \
       -batch_size 32
