import json
import codecs
import sys
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

pattern = re.compile('[\W]*')


def ans_tag(answer, corpus):
    src = []
    porter_stemmer, stopwords_eng = PorterStemmer(), stopwords.words('english')
    for index, sample in tqdm(enumerate(zip(answer, corpus))):
        ans, cps = sample[0], sample[1]
        ans, nodes, edges = ans.strip().split(), cps['nodes'], cps['edges']
        node_words = [node['word'].split(' ') for node in nodes]
        node_stem = [[porter_stemmer.stem(w) for w in node] for node in node_words]
        ans = [w for w in ans if len(w) > 0 and w not in stopwords_eng and not pattern.fullmatch(w)]
        ans_stem = [porter_stemmer.stem(w) for w in ans]
        src.append(cps['text'])
        ans_indexes = [[] for _ in ans]
        node_contain = [0 for _ in nodes]
        for idx, word in enumerate(ans):
            for id_node, words in enumerate(node_words):
                if ans_stem[idx] in node_stem[id_node] or word in words or any([w.count(word) > 0 for w in words]):
                    ans_indexes[idx].append(id_node)
                    node_contain[id_node] += 1
        for indexes in ans_indexes:
            if len(indexes) > 0:
                indexes.sort(key=lambda idx: (node_contain[idx], -len(nodes[idx]['index'])))
                nodes[indexes[-1]]['ans'] = 1
                for idx in indexes[:-1]:
                    if edges[idx][indexes[-1]] == 'CHILD':
                        nodes[idx]['ans'] = 1
        for idx, node in enumerate(nodes):
            if 'ans' not in node:
                nodes[idx]['ans'] = 0
        #for idx, node in enumerate(nodes):
            #words = node['word'].strip().split()
            #flag = any([w in words for w in ans])
            #if flag:
                #nodes[idx]['type'] = 'A'
        corpus[index]['nodes'] = nodes
    return src, corpus


if __name__ == '__main__':
    answer = sys.argv[1]
    corpusf = sys.argv[2]
    source = sys.argv[3]

    with open(answer, 'r', encoding='utf-8') as f:
        ans = f.read().strip().split('\n')

    corpus = json_load(corpusf)

    src, corpus = ans_tag(answer, corpus)

    with open(source, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src) + '\n')
    
    json_dump(corpus, corpusf)
    
