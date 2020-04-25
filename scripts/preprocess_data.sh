#!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src

cd ${EXEHOME}

python preprocess.py \
       -train_src ${DATAHOME}/text-data/train.src.txt -train_tgt ${DATAHOME}/text-data/train.tgt.txt \
       -valid_src ${DATAHOME}/text-data/valid.src.txt -valid_tgt ${DATAHOME}/text-data/valid.tgt.txt \
       -train_ans ${DATAHOME}/text-data/train.ans.txt -valid_ans ${DATAHOME}/text-data/valid.ans.txt \
       -train_graph ${DATAHOME}/json-data/train.tag.json -valid_graph ${DATAHOME}/json-data/valid.tag.json \
       -node_feature \
       -copy \
       -answer \
       -save_sequence_data ${DATAHOME}/preprocessed-data/preprcessed_sequence_data.pt \
       -save_graph_data ${DATAHOME}/preprocessed-data/preprcessed_graph_data.pt \
       -train_dataset ${DATAHOME}/Datasets/train_dataset.pt \
       -valid_dataset ${DATAHOME}/Datasets/valid_dataset.pt \
       -src_seq_length 200 -tgt_seq_length 50 \
       -src_vocab_size 50000 -tgt_vocab_size 50000 \
       -src_words_min_frequency 3 -tgt_words_min_frequency 2 \
       -vocab_trunc_mode frequency \
       -pre_trained_vocab ${GLOVEHOME}/glove.840B.300d.txt -word_vec_size 300 \
       -batch_size 32
