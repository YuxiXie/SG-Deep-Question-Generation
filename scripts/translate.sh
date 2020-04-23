 #!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src
MODELHOME=${HOME}/models/generator
LOGHOME=${HOME}/logs/predictions

mkdir -p ${LOGHOME}

cd ${EXEHOME}

python translate.py \
       -model ${DATAHOME}/generator_15.12441_bleu4.chkpt \
       -sequence_data ${DATAHOME}/preprcessed_sequence_data.pt \
       -graph_data ${DATAHOME}/preprcessed_graph_data.pt \
       -valid_data ${DATAHOME}/Dataset/valid_dataset.pt \
       -output ${LOGHOME}/prediction.txt \
       -beam_size 5 \
       -batch_size 16 \
       -gpus 0