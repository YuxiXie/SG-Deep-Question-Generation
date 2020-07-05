import sys
from tqdm import tqdm

import json
import codecs


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def add_evidence(key, dictionary, evidence, evidence_index, context):
    if key[0] in dictionary:
        try:
            evidence.append([dictionary[key[0]][key[1]]])
            evidence_index.append([key[0], (key[1], key[1] + 1)])
        except:
            print("ERROR 1")
            evidence.append(dictionary[key[0]])
            evidence_index.append([key[0], (0, len(dictionary[key[0]]))])
        context[key[0]] = dictionary[key[0]]
    else:
        print("ERROR 2")
        flags = [k.count(key[0]) for k in dictionary]
        if any(flags):
            index = flags.index(True)
            key[0] = list(dictionary.keys())
            key[0] = key[0][index]
            try:
                evidence.append([dictionary[key[0]][key[1]]])
                evidence_index.append([key[0], (key[1], key[1] + 1)])
            except:
                print("ERROR 4")
                evidence.append(dictionary[key[0]])
                evidence_index.append([key[0], (0, len(dictionary[key[0]]))])
            context[key[0]] = dictionary[key[0]]
        else:
            print("ERROR 3")
            return False
    return True


def extract(data):
    paragraphs = [sample['context'] for sample in data]
    paragraphs = {c[0]:c[1] for sample in paragraphs for c in sample}

    corpus = []
    for sample in tqdm(data, desc='   - (Extract information) -   '):
        context = {c[0]:c[1] for c in sample['context']}
        supporting_facts = sample['supporting_facts']
        answer = sample['answer']
        question = sample['question']

        evidence, evidence_index = [], []
        ctxt = {}

        for sf in supporting_facts:
            if not add_evidence(sf, context, evidence, evidence_index, ctxt):
                add_evidence(sf, paragraphs, evidence, evidence_index, ctxt)

        if len(evidence) > 0:
            evidence = [{'text':evd, 'index':idx} for evd, idx in zip(evidence, evidence_index)]
            sample = {'question':question, 'answer': answer, 'evidence':evidence, 'context':ctxt}
            corpus.append(sample)

    return corpus


def overlap(corpus):
    train, valid = [], []
    questions, sources = [], []
    for sample in tqdm(corpus['train'], desc='   - (Deal with overlapping) -   '):
        if sample['question'] not in questions:
            questions.append(sample['question'])
            train.append(sample)
            try:
                sources.append('\t'.join(['\t'.join(sent['text']) for sent in sample['evidence']]))
            except:
                import ipdb; ipdb.set_trace()
    for sample in tqdm(corpus['valid'], desc='   - (Deal with overlapping) -   '):
        tmp = '\t'.join(['\t'.join(sent['text']) for sent in sample['evidence']])
        if tmp not in sources:
            valid.append(sample)
    print(len(train), len(valid))
    return {'train':train, 'valid':valid}


def process(train, valid):
    corpus = {'train':extract(train), 'valid':extract(valid)}
    corpus = overlap(corpus)
    return corpus


if __name__ == '__main__':
    ##=== load raw HotpotQA dataset ===##
    train = json_load(sys.argv[1])
    valid = json_load(sys.argv[2])
    ##=== run processing ===##
    corpus = process(train, valid)
    ##=== directory for saving train & valid data ===##
    json_dump(corpus['train'], sys.argv[3] + 'data.train.json')
    json_dump(corpus['valid'], sys.argv[3] + 'data.valid.json')
