 #!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src
MODELHOME=${HOME}/models
LOGHOME=${HOME}/predictions

mkdir -p ${LOGHOME}

cd ${EXEHOME}

python translate.py \
       -model ${MODELHOME}/generator_15.12441_bleu4.chkpt \
       -sequence_data ${DATAHOME}/preprocessed-data/preprcessed_sequence_data.pt \
       -graph_data ${DATAHOME}/preprocessed-data/preprcessed_graph_data.pt \
       -valid_data ${DATAHOME}/Datasets/valid_dataset.pt \
       -output ${LOGHOME}/prediction.txt \
       -beam_size 5 \
       -batch_size 16 \
       -gpus 0
