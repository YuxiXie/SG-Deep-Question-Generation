# Semantic Graphs for Generating Deep Questions

This repository contains code and models for the paper: [Semantic Graphs for Generating Deep Questions (ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.135/). Below is the framework of our proposed model (on the right) together with an input example (on the left).

![Model Framework](model.jpg)

## Requirements

#### Environment

```
allennlp 1.0.0
allennlp-models 1.0.0

pytorch 1.4.0
nltk 3.4.4
numpy 1.18.1
tqdm 4.32.2
```

#### Data Preprocessing

We release [all the datasets below](https://drive.google.com/drive/folders/1uPQaK-cWcbkZapmC3qROkmddC_st5uhv?usp=sharing) which are processed based on [HotpotQA](https://hotpotqa.github.io/). 

1. get tokenized data files of `documents`, `questions`, `answers`

	* get results in folder [`text-data`](https://drive.google.com/drive/folders/1nhBfk2EvOHGDRq6vPCf8Pk8wZFL0dqbf?usp=sharing)

2. prepare the json files ready as illustrated in [`build-semantic-graphs`](https://github.com/YuxiXie/SG-Deep-Question-Generation/tree/master/build-semantic-graphs)

	*  get results in folder [`json-data`](https://drive.google.com/drive/folders/10idPzICLR_OhEZHfGnvgZcqAB1x509mE?usp=sharing)

3. run [`scripts/preprocess_data.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/preprocess_data.sh) to get the preprocessed data ready for training

	* get results in folder [`preprocessed-data`](https://drive.google.com/drive/folders/1ZvMRDtb5EeEaylEC-pKSLID0COArJ6Nf?usp=sharing)
	
	* utilize `glove.840B.300d.txt` from [GloVe](https://nlp.stanford.edu/projects/glove/) to initialize the word-embeddings

#### Models

We release both classifier and generator models in this work. The models are constructed based on a ***sequence-to-sequence*** architecture. Typically, we use ***GRU*** and ***GNN*** in the encoder and ***GRU*** in the decoder, you can choose other methods (*e.g.* ***Transformer***) which have also been implemented in our repository.

* [classifier](https://drive.google.com/file/d/1X_fdQgQ1yv15e7QCOXkhbWpYLnoT80mH/view?usp=sharing): accuracy - 84.06773%

* [generator](https://drive.google.com/file/d/1Fck0qVYNnLLz3f815CinRfWFrO2ceIfI/view?usp=sharing): BLeU-4 - 15.28304

## Training

* run [`scripts/train_classifier.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/train_classifier.sh) to train on the ***Content Selection*** task

* run [`scripts/train_generator.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/train_generator.sh) to train on the ***Question Generation*** task, the default one is to finetune based on the pretrained classifier

## Translating

* run [`scripts/translate.sh`](https://github.com/YuxiXie/SG-Deep-Question-Generation/blob/master/scripts/translate.sh) to get the prediction on the validation dataset

## Evaluating

We take use of the [Evaluation codes for MS COCO caption generation](https://github.com/salaniz/pycocoevalcap) for evaluation on automatic metrics.

  - To install pycocoevalcap and the pycocotools dependency, run:

```
pip install git+https://github.com/salaniz/pycocoevalcap
```

  - To evaluate the results in the translated file, _e.g._ `prediction.txt`, run:

```
python evaluate_metrics.py prediction.txt
```

## Citation
```
    @inproceedings{pan-etal-2020-DQG,
      title = {Semantic Graphs for Generating Deep Questions},
      author = {Pan, Liangming and Xie, Yuxi and Feng, Yansong and Chua, Tat-Seng and Kan, Min-Yen},
      booktitle = {Proceedings of Annual Meeting of the Association for Computational Linguistics (ACL)},
      year = {2020}
    }
```
