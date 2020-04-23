"""Question Tagging
Generate groudtruth for Context Selection
"""

import sys
from tqdm import tqdm

import json
import codecs

import nltk
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords

import re

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


pattern = re.compile('[\W]*')
verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG', 'IN', 'TO', 'PP']
noun_pos = ['NN', 'NNP', 'NNS', 'NNPS']
modifier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']


def text_load(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    return data


def tag(nodes, edges, question):
    porter_stemmer, stopwords_eng = PorterStemmer(), stopwords.words('english')
    ## get words and word-stems
    words, question = [node['word'].split(' ') for node in nodes], [w for w in question.split(' ') if len(w) > 0]
    node_stem, question_stem = [[porter_stemmer.stem(w) for w in sent] for sent in words], [porter_stemmer.stem(w) for w in question]
    ## search for the node list covering the question by each word
    question_index, node_contain = [[] for _ in question], [0 for _ in nodes]
    for index, word in enumerate(question):
        if len(word) > 0 and word not in stopwords_eng and not pattern.fullmatch(word):
            for idx, node_words in enumerate(words):
                if question_stem[index] in node_stem[idx] or word in node_words or any([w.count(word) > 0 for w in node_words]):
                    question_index[index].append(idx)
                    node_contain[idx] += 1
    ## tag the node which covers more in the question
    for index in question_index:
        if len(index) > 0:
            index.sort(key=lambda idx: (node_contain[idx], -len(nodes[idx]['index'])))
            nodes[index[-1]]['tag'] = 1
    ## tag the node which has 'SIMILAR' edge
    #  (we assume this kind of nodes are important for asking questions)
    for index, node in enumerate(nodes):
        if 'tag' not in node:
            nodes[index]['tag'] = 1 if 'SIMILAR' in edges[index] else 0
    
    return nodes


def main(raw, questions):
    for idx, sample in tqdm(enumerate(zip(raw, questions)), desc='     (TAGGING)     '):
        sample, question = sample[0], sample[1]
        nodes, edges = sample['nodes'], sample['edges']
        raw[idx]['nodes'] = tag(nodes, edges, question)
    return raw


if __name__ == '__main__':
    ##=== load files ===##
    data = json_load(sys.argv[1])
    questions = text_load(sys.argv[2])
    ##=== tagging ===##
    new = main(data, questions)
    ##=== dump file ===##
    json_dump(new, sys.argv[3])