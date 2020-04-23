# Semantic Graphs for Generating Deep Questions

This repository contains code and models for the paper: [Semantic Graphs for Generating Deep Questions (ACL 2020)]().

## Requirements

#### Data Preprocessing

We release all the datasets below which are processed based on [HotpotQA](https://hotpotqa.github.io/).

1. get tokenized data files of `documents`, `questions`, `answers`, and the results of ***Dependency Parsing*** and ***Coreference Resolution***

2. prepare the json files ready as illustrated in [`build-semantic-graphs`](https://github.com/YuxiXie/SG-Deep-Question-Generation/tree/master/build-semantic-graphs)

3. run [`scripts/preprocess_data.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/preprocess_data.sh) to get the preprocessed data ready for training

#### Models

We release both classifier and generator models in this work. The models are constructed based on a ***sequence-to-sequence*** architecture. Typically, we use ***GRU*** and ***GNN*** in the encoder and ***GRU*** in the decoder, you can choose other methods (*e.g.* ***Transformer***) which have also been implemented in our repository.

* [classifier](): accuracy - 84.32607%

* [generator](): BLeU-4 - 15.12441

## Training

* run [`scripts/train_classifier.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/train_classifier.sh) to train on the ***Content Selection*** task

* run [`scripts/train_generator.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/train_generator.sh) to train on the ***Question Generation*** task, the default one is finetuning based on the pretrained classifier

## Translating / Testing

* run [`scripts/translate.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/translate.sh) to get the prediction file on the validation dataset

## Citation
```
    @article{pan2019sgdqg,
      title={Semantic Graphs for Generating Deep Questions},
      author={Liangming Pan and Yuxi Xie and Yansong Feng and Tat-Seng Chua and Min-Yen Kan},
      journal={ACL 2020},
      year={2020}
    }
```
