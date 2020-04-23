"""
In this code, we do node pruning and merging in each tree.

1st prune: we prune nodes which represent unimportant information
    e.g. punctuation, conjunction
    
2nd merge: merge modifier-child nodes into their parent nodes
"""

import sys
from tqdm import tqdm

import json
import codecs

import re
from nltk.corpus import stopwords


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


prep_pos = ['PP', 'IN', 'TO']
modefier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
modifiers = ['amod', 'nn', 'mwe', 'num', 'quantmod', 'dep', 'number', 'auxpass', 'partmod', 'poss', 
             'possessive', 'neg', 'advmod', 'npadvmod', 'advcl', 'aux', 'det', 'predet', 'appos']
prune_list = ['punct', 'cc', 'preconj']


def merge(node, sequence):
    indexes = [idx for idx in node['index']]  

    if 'attribute' in node and node['dep'] not in ['det', 'punct']:
        attr1, attr2 = [], []
        for a in node['attribute']:
            if ('attribute' in a) or ('noun' in a) or ('verb' in a):
                attr2.append(a)
            elif a['dep'] in modifiers or a['pos'] in modefier_pos:
                attr1.append(a)
                indexes += [idx for idx in a['index']]
            else:
                attr2.append(a)

        if len(attr1) > 0:
            indexes.sort(key=lambda x:x, reverse=False)
            node['word'], node['index'] = [sequence[i] for i in indexes], indexes
            if len(attr2) > 0:
                node['attribute'] = [a for a in attr2]
            else:
                del node['attribute']

    return node


def prune(node, sequence):
    ## collect child nodes
    nouns = node['noun'] if 'noun' in node else []
    verbs = node['verb'] if 'verb' in node else []
    attributes = node['attribute'] if 'attribute' in node else []
    ## prune and update child node sets
    Ns, Vs, As = [], [], []
    for child in nouns + verbs + attributes:
        if child['pos'] not in prep_pos and child['dep'] in prune_list:
            Ns += child['noun'] if 'noun' in child else []
            Vs += child['verb'] if 'verb' in child else []
            As += child['attribute'] if 'attribute' in child else []
        else:
            Ns += [child] if child in nouns else []
            Vs += [child] if child in verbs else []
            As += [child] if child in attributes else []
    ## do pruning and merging on child nodes
    Ns = [prune(n, sequence) for n in Ns]
    Vs = [prune(v, sequence) for v in Vs]
    As = [prune(a, sequence) for a in As]
    ## do merging
    slf = {k:v for k,v in node.items() if k not in ['noun', 'verb', 'attribute']}
    if As:
        slf['attribute'] = As
    slf = merge(slf, sequence)
    ## get final node
    wrap = {'dep':slf['dep'], 'word':slf['word'], 'index':slf['index'], 'pos':slf['pos'], 
            'type':slf['type'], 'noun':Ns, 'verb':Vs}
    if 'attribute' in slf:
        wrap['attribute'] = slf['attribute']    
    if not Ns:
        del wrap['noun']
    if not Vs:
        del wrap['verb']        
    return wrap


if __name__ == '__main__':
    ##=== load raw file ===##
    raw = json_load(sys.argv[1])
    ##=== prune and merge trees ===##
    tree = []
    for sample in tqdm(raw, desc='   - (Pruning Trees) -   '):
        evidence = [{'sequence':sent['words'], 'graph':prune(sent['tree'], sent['words'])} for sent in sample]
        tree.append(evidence)
    ##=== dump file ===##
    json_dump(tree, sys.argv[2])
