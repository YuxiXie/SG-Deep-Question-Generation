import sys
import codecs
import json
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def get_context(data):
    context = {}
    for sample in tqdm(data):
        for k, v in sample['context'].items():
            context[k] = v
    return context


def get_dependency(sent, dependency_parser):
    if len(sent.strip()) == 0:
        return None
    try:
        sent = dependency_parser.predict(sentence=sent)
    except:
        import ipdb; ipdb.set_trace()
    words, pos, heads, dependencies = sent['words'], sent['pos'], sent['predicted_heads'], sent['predicted_dependencies']
    result = [{'word':w, 'pos':p, 'head':h - 1, 'dep':d} for w, p, h, d in zip(words, pos, heads, dependencies)]
    return result


def dependency_parse(raw, filename):
    dependency_parser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    context = {
        key: [
            get_dependency(sent, dependency_parser) for sent in value
        ] for key, value in tqdm(raw.items(), desc='   - (Dependency Parsing: 1st) -   ')
    }
    json_dump(context, filename)


def get_coreference(doc, coref_reslt, pronouns, title):

    def get_crf(span, words):
        phrase = []
        for i in range(span[0], span[1] + 1):
            phrase += [words[i]]
        return (' '.join(phrase), span[0], span[1] - span[0] + 1)

    def get_best(crf):
        crf.sort(key=lambda x: x[2], reverse=True)
        if crf[0][2] == 1:
            crf.sort(key=lambda x: len(x[0]), reverse=True)
        for w in crf:
            if w[0].lower() not in pronouns and w[0].lower() != '\t':
                return w[0]
        return None

    doc = coref_reslt.predict(document=doc)
    words = [w.strip(' ') for w in doc['document']]
    clusters = doc['clusters']

    for group in clusters:
        crf = [get_crf(span, words) for span in group]
        entity = get_best(crf)
        if entity in ['\t', None]:
            try:
                entity = coref_reslt.predict(document=title)
                entity = ' '.join(entity['document'])
            except:
                entity = ' '.join(word_tokenize(title))
        if entity not in ['\t', None]:
            for phrase in crf:
                if phrase[0].lower() in pronouns:
                    index = phrase[1]
                    words[index] = entity

    doc, sent = [], []
    for word in words:
        if word.strip(' ') == '\t':
            doc.append(sent)
            sent = []
        else:
            if word.count('\t'):
                print(word)
                word = word.strip('\t')
            sent.append(word)
    doc.append(sent)
    return doc


def coreference_resolution(raw, filename):
    pronouns = ['it', 'its', 'he', 'him', 'his', 'she', 'her', 'they', 'their', 'them']
    raw = {k: '\t'.join(v) for k,v in raw.items()}
    coref_reslt = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    context = {
        key: get_coreference(value, coref_reslt, pronouns, key) for key, value in tqdm(raw.items(), desc='  - (crf for evidence) ')
    }
    json_dump(context, filename)


def get_ner(doc, ner_tagger):
    try:
        doc = ner_tagger.predict(sentence=doc)
    except:
        return [[doc, 'O']]
    words, tags = doc['words'], doc['tags']
    return [[w, t] for w, t in zip(words, tags)]


def ner_tag(raw, filename):
    ner_tagger = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    raw = [[d[0], d[0]] for d in raw]    #raw = [[d[0], '\t'.join(d[1])] for d in raw]
    context = {sample[0]: get_ner(sample[1], ner_tagger) for sample in tqdm(raw, desc='  - (ner for evidence) ')}
    json_dump(context, filename)


def sr_labeling(sent, sr_labeler):
    if len(sent.strip()) == 0:
        return None
    try:
        sent = sr_labeler.predict(sentence=sent)
    except:
        import ipdb; ipdb.set_trace()
    length, words, verbs = len(sent['words']), sent['words'], sent['verbs']
    tags = [verb['tags'] for verb in verbs]
    return {'words':words, 'tags':tags}


def semantic_role_labeling(raw, filename):
    sr_labeler = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    context = {sample[0]: [sr_labeling(sent, sr_labeler) for sent in sample[1]] for sample in tqdm(raw, desc='   - (Semantic Role Labeling: 1st) -   ')}
    json_dump(context, filename)


if __name__ == '__main__':
    train_data_file, valid_data_file = sys.argv[1], sys.argv[2]
    data = json_load(train_data_file) + json_load(valid_data_file)
    context = get_context(data)
    print('number of context:', len(context))

    dependency_parse(context, sys.argv[3])
    coreference_resolution(context, sys.argv[4])

    # ner_tag(context, sys.argv[3])
    # semantic_role_labeling(context, sys.argv[3])
