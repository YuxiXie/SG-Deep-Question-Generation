"""
In this code, we merge the graphs of the sentences in the same evidence into a unified graph.

To connect between subgraphs, we introduce 'SIMILAR' edges
"""

import sys
from tqdm import tqdm

import nltk
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords

import json
import codecs
import re


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


pattern = re.compile('[\W]*')
verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG', 'IN', 'TO', 'PP']
noun_pos = ['NN', 'NNP', 'NNS', 'NNPS']
modifier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj']
modifiers = ['amod', 'nn', 'mwe']
pronouns = ['it', 'its', 'him', 'he', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']


def draw_edge(node_raw_id, final_nodes, raw_nodes, final_edges, raw_edges,
              accumulate_node, accumulate_word, reindex_list):
    stopwords_eng = stopwords.words('english')

    NDE, raw_node_num = raw_nodes[node_raw_id], len(raw_nodes)
    NDE_words = NDE['word'].strip().split(' ')

    final_nodes[accumulate_node + node_raw_id]['index'] = [ii + accumulate_word for i in NDE['index'] for ii in reindex_list[i]]

    ## copy existed edges
    for i, edge in enumerate(raw_edges[node_raw_id]):
        if i == node_raw_id:
            final_edges[accumulate_node + node_raw_id][accumulate_node + node_raw_id] = 'SELF'
        elif edge:
            final_edges[accumulate_node + node_raw_id][accumulate_node + i] = edge
    ## connect 'SIMILAR' nodes
    for i, node in enumerate(final_nodes[accumulate_node + raw_node_num: ]):
        i_idx = i + accumulate_node + raw_node_num
        words = node['word'].strip().split(' ')
        # get 'important' word list
        word_i = [w for w in words if len(w) > 0 and w not in stopwords_eng and not pattern.fullmatch(w) and w.lower() not in pronouns]
        word_j = [w for w in NDE_words if len(w) > 0 and w not in stopwords_eng and not pattern.fullmatch(w) and w.lower() not in pronouns]
        # get common 'important' words
        common1 = [w for w in word_i if w in word_j or w.lower() in [ww.lower() for ww in word_j]]
        common2 = [w for w in word_j if w in word_i or w.lower() in [ww.lower() for ww in word_i]]
        common = common1 if len(common1) < len(common2) else common2
        # whether have noun-or-modifier-like words in common
        mono_pos = nltk.pos_tag(common)
        flag_pos = any([mono[1] in noun_pos + modifier_pos for mono in mono_pos])
        # whether have upper-case words in common
        flag_up = any([not w.islower() for w in common])
        # whether is of the same kind of pos&dep tag
        pos_qualify = NDE['pos'] in verb_pos + noun_pos and node['pos'] in verb_pos + noun_pos
        dep_qualify = NDE['dep'] in subj_and_obj + modifiers and node['dep'] in subj_and_obj + modifiers
        
        if pos_qualify or dep_qualify:
            if (flag_up or flag_pos) and len(word_i) * len(word_j) > 0:
                prb1, prb2 = len(common1) / len(word_i), len(common2) / len(word_j)
                if max(prb1, prb2) > 1/2 and min(prb1, prb2) > 1/3:     # requirement of the lavel of overlapping
                    final_edges[accumulate_node + node_raw_id][i_idx] = final_edges[i_idx][accumulate_node + node_raw_id] = 'SIMILAR'


def merge(corpus):

    def reindex(sequence):
        '''do reindexing
        because we have coreference resolution, which means
        there may be more than one words in the so-called 'one' word in fact
        '''
        cnt, new_seq = 0, []
        dicts = [[] for _ in sequence]
        for i, w in enumerate(sequence):
            wrd_cnt = max(len(w.strip().split(' ')), 1)
            dicts[i] = [i for i in range(cnt, cnt + wrd_cnt)]
            cnt += wrd_cnt
            new_seq += w.strip().split(' ')
        length = len(new_seq)
        return dicts, length, new_seq

    ## initialize node and edge list
    sequences, subgraphs = [sent['sequence'] for sent in corpus], [sent['graph'] for sent in corpus]

    final_nodes = [node for subgraph in subgraphs for node in subgraph['nodes']]
    nodes_num = len(final_nodes)
    final_edges = [['' for _ in range(nodes_num)] for _ in range(nodes_num)]

    ## merge subgraphs into final graph
    word_cnt, node_cnt, new_sequences = 0, 0, []
    for sequence, subgraph in zip(sequences, subgraphs):
        indexes, length, new_seq = reindex(sequence)
        nodes, edges = subgraph['nodes'], subgraph['edges']
        for idx, node in enumerate(nodes):
            draw_edge(node_raw_id=idx, final_nodes=final_nodes, raw_nodes=nodes, 
                      final_edges=final_edges, raw_edges=edges, accumulate_node=node_cnt, 
                      accumulate_word=word_cnt, reindex_list=indexes)
        
        node_cnt += len(nodes)
        word_cnt += length
        new_sequences += new_seq
    
    sequences = ' '.join(new_sequences)

    return {'nodes':final_nodes, 'edges':final_edges, 'text':sequences}


if __name__ == '__main__':
    ##=== load file ===##
    raw = json_load(sys.argv[1])
    ##=== merge graphs ===##
    graphs = []
    for sample in tqdm(raw, desc='   - (Building Graphs) -   '):
        graph = merge(sample)
        graphs.append(graph)
    ##=== dump file ===##
    json_dump(graphs, sys.argv[2])