import math
import json
import codecs
import torch
import argparse
from tqdm import tqdm

import pargs
import onqg.dataset.Constants as Constants
from onqg.dataset import Vocab
from onqg.dataset import Dataset


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))


def load_vocab(filename):
    vocab_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    text = [word.split(' ') for word in text]
    vocab_dict = {word[0]:word[1:] for word in text}
    vocab_dict = {k:[float(d) for d in v] for k,v in vocab_dict.items()}
    return vocab_dict


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    data = data.split('\n')
    data = [sent.split(' ') for sent in data]
    return data


def filter_data(data_list, opt):
    data_num = len(data_list)
    idx = list(range(len(data_list[0])))
    rst = [[] for _ in range(data_num)]

    final_indexes = []    
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        src_len = len(src)
        if src_len <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length:
            if len(src) * len(tgt) > 0 and src_len >= 10:  # TODO: fix this magic number
                rst[0].append(src)
                rst[1].append([Constants.BOS_WORD] + tgt + [Constants.EOS_WORD])
                final_indexes.append(i)
                for j in range(2, data_num):
                    sent = data_list[j][i]
                    rst[j].append(sent)

    print("change data size from " + str(len(idx)) + " to " + str(len(rst[0])))
    return rst, final_indexes


def convert_word_to_idx(text, vocab, lower=True, sep=False, pretrained=''):

    def lower_sent(sent):
        for idx, w in enumerate(sent):
            if w not in [Constants.BOS_WORD, Constants.EOS_WORD, Constants.PAD_WORD, Constants.UNK_WORD, Constants.SEP_WORD]:
                sent[idx] = w.lower()
        return sent
    
    def get_dict(sent, length, raw, separate=False):
        raw = raw.split(' ')
        bert_sent = [w for w in sent]
        ans_indexes = []
        if separate:
            sep_id = sent.index(Constants.SEP_WORD)
            sent = sent[:sep_id]
            ans_indexes = [i for i in range(sep_id + 1, len(bert_sent) - 1)]

        indexes = [[i for i in ans_indexes] for _ in raw]
        word, idraw = '', 0
        for idx, w in enumerate(sent):
            if word == raw[idraw] and idx != 0:
                idraw += 1
                while len(raw[idraw]) < len(w):
                    idraw += 1
                word = w
            else:
                word = word + w.lstrip('##')
                while len(raw[idraw]) < len(word):
                    idraw += 1
            indexes[idraw].append(idx)
            
        flags = [len(idx) > 0 for idx in indexes]
        return indexes

    if pretrained.count('bert'):
        lengths = [len(sent) for sent in text]
        text = [' '.join(sent) for sent in text]
        if sep:
            text = [sent.split(' ' + Constants.SEP_WORD + ' ') for sent in text]
            tokens = [vocab.tokenizer.tokenize(sent[0]) + [Constants.SEP_WORD] + vocab.tokenizer.tokenize(sent[1]) + [Constants.SEP_WORD] for sent in text]
            text = [sent[0] for sent in text]
            index_dict = [get_dict(sent, length, raw.lower(), separate=sep) for sent, length, raw in zip(tokens, lengths, text)]
        else:
            tokens = [vocab.tokenizer.tokenize(sent) for sent in text]
            index_dict = [get_dict(sent, length, raw.lower()) for sent, length, raw in zip(tokens, lengths, text)]
    else:
        index_dict = None
        tokens = [lower_sent(sent) for sent in text] if lower else text    

    indexes = [vocab.convertToIdx(sent) for sent in tokens]

    return indexes, tokens, index_dict


def get_embedding(vocab_dict, vocab):

    def get_vector(idx):
        word = vocab.idxToLabel[idx]
        if idx in vocab.special or word not in vocab_dict:
            vector = torch.tensor([])
            vector = vector.new_full((opt.word_vec_size,), 1.0)
            vector.normal_(0, math.sqrt(6 / (1 + vector.size(0))))
        else:
            vector = torch.Tensor(vocab_dict[word])
        return vector
    
    embedding = [get_vector(idx) for idx in range(vocab.size)]
    embedding = torch.stack(embedding)
    
    print(embedding.size())

    return embedding


def get_data(files, opt):
    src, tgt = load_file(files['src']), load_file(files['tgt'])
    data_list = [src, tgt]
    if opt.answer:
        data_list.append(load_file(files['ans']))
    if opt.feature:
        data_list += [load_file(filename) for filename in files['feats']]
    
    data_list, final_indexes = filter_data(data_list, opt)

    rst = {'src':data_list[0], 'tgt':data_list[1]}
    i = 2
    if opt.answer:
        rst['ans'] = data_list[i]
        i += 1
    if opt.feature:
        rst['feats'] = [data_list[i] for i in range(i, i + len(files['feats']))]
        i += 1     
    return rst, final_indexes


def merge_ans(src, ans):
    rst = [s + [Constants.SEP_WORD] + a + [Constants.SEP_WORD] for s, a in zip(src, ans)]
    return rst


def wrap_copy_idx(splited, tgt, tgt_vocab, bert, vocab_dict):
    
    def map_src(sp):
        sp_split = {}
        if bert:
            tmp_idx = 0
            tmp_word = ''
            for i, w in enumerate(sp):
                if not w.startswith('##'):
                    if tmp_word:
                        sp_split[tmp_word] = tmp_idx
                    tmp_word = w
                    tmp_idx = i
                else:
                    tmp_word += w.lstrip('##')
            sp_split[tmp_word] = tmp_idx
        else:
            sp_split = {w:idx for idx, w in enumerate(sp)}
        return sp_split

    def wrap_sent(sp, t):
        sp_dict = map_src(sp)
        swt, cpt = [0 for w in t], [0 for w in t]
        for i, w in enumerate(t):
            # if w not in tgt_vocab.labelToIdx or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
            if w not in tgt_vocab.labelToIdx or w not in vocab_dict or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
                if w in sp_dict:
                    swt[i] = 1
                    cpt[i] = sp_dict[w]
        return torch.Tensor(swt), torch.LongTensor(cpt)
    
    copy = [wrap_sent(sp, t) for sp, t in zip(splited, tgt)]
    switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
    return [switch, cp_tgt]


def sequence_data(opt):
    #========== get data ==========#
    train_files = {'src':opt.train_src, 'tgt':opt.train_tgt}
    valid_files = {'src':opt.valid_src, 'tgt':opt.valid_tgt}
    if opt.answer:
        assert opt.train_ans and opt.valid_ans, "Answer files of train and valid must be given"
        train_files['ans'], valid_files['ans'] = opt.train_ans, opt.valid_ans
    if opt.feature:
        assert len(opt.train_feats) == len(opt.valid_feats) and len(opt.train_feats) > 0
        train_files['feats'], valid_files['feats'] = opt.train_feats, opt.valid_feats
    
    train_data, train_final_indexes = get_data(train_files, opt)
    valid_data, valid_final_indexes = get_data(valid_files, opt)

    train_src, train_tgt = train_data['src'], train_data['tgt']
    valid_src, valid_tgt = valid_data['src'], valid_data['tgt']
    train_ans = train_data['ans'] if opt.answer else None
    valid_ans = valid_data['ans'] if opt.answer else None
    train_feats = train_data['feats'] if opt.feature else None
    valid_feats = valid_data['feats'] if opt.feature else None

    #========== build vocabulary ==========#
    print('Loading pretrained word embeddings ...')
    pre_trained_vocab = load_vocab(opt.pre_trained_vocab) if opt.pre_trained_vocab else None  
    print('Done .')
    
    if opt.share_vocab:
        assert not opt.pretrained
        print("build src & tgt vocabulary")
        corpus = train_src + train_tgt
        options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'tgt':True,
                   'size':max(opt.src_vocab_size, opt.tgt_vocab_size),
                   'frequency':min(opt.src_words_min_frequency, opt.tgt_words_min_frequency)}
        vocab = Vocab.from_opt(corpus=corpus, opt=options)
        src_vocab = tgt_vocab = vocab
    else:
        print("build src vocabulary")
        if opt.pretrained:
            options = {'separate':False, 'tgt':False, 'lower':True}
            src_vocab = Vocab.from_opt(pretrained=opt.pretrained, opt=options)
        else:
            corpus = train_src + train_ans if opt.answer else train_src
            options = {'lower':True, 'mode':'size', 'tgt':False, 
                       'size':opt.src_vocab_size, 'frequency':opt.src_words_min_frequency}
            src_vocab = Vocab.from_opt(corpus=corpus, opt=options)
            ans_vocab = src_vocab if opt.answer else None
        
        print("build tgt vocabulary")
        options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'tgt':True, 
                   'size':opt.tgt_vocab_size, 'frequency':opt.tgt_words_min_frequency}
        tgt_vocab = Vocab.from_opt(corpus=train_tgt, opt=options)
    
    options = {'lower':False, 'mode':'size', 'size':opt.feat_vocab_size, 
               'frequency':opt.feat_words_min_frequency, 'tgt':False}
    feats_vocab = [Vocab.from_opt(corpus=feat, opt=options) for feat in train_feats] if opt.feature else None
        
    #========== word to index ==========#
    train_src_idx, train_src_tokens, train_src_indexes = convert_word_to_idx(train_src, src_vocab, sep=opt.answer == 'sep',
                                                                             pretrained=opt.pretrained)
    valid_src_idx, valid_src_tokens, valid_src_indexes = convert_word_to_idx(valid_src, src_vocab, sep=opt.answer == 'sep', 
                                                                             pretrained=opt.pretrained)
    train_tgt_idx, train_tgt_tokens, _ = convert_word_to_idx(train_tgt, tgt_vocab)    
    valid_tgt_idx, valid_tgt_tokens, _ = convert_word_to_idx(valid_tgt, tgt_vocab)

    train_copy = wrap_copy_idx(train_src_tokens, train_tgt_tokens, tgt_vocab, opt.pretrained, pre_trained_vocab) if opt.copy else [None, None]
    valid_copy = wrap_copy_idx(valid_src_tokens, valid_tgt_tokens, tgt_vocab, opt.pretrained, pre_trained_vocab) if opt.copy else [None, None]
    train_copy_switch, train_copy_tgt = train_copy[0], train_copy[1]
    valid_copy_switch, valid_copy_tgt = valid_copy[0], valid_copy[1]

    train_ans_idx = convert_word_to_idx(train_ans, ans_vocab)[0] if opt.answer else None
    valid_ans_idx = convert_word_to_idx(valid_ans, ans_vocab)[0] if opt.answer else None
    
    train_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab 
                        in zip(train_feats, feats_vocab)] if opt.feature else None
    valid_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab 
                        in zip(valid_feats, feats_vocab)] if opt.feature else None

    #========== prepare pretrained vetors ==========#
    if pre_trained_vocab:
        pre_trained_src_vocab = None if opt.pretrained else get_embedding(pre_trained_vocab, src_vocab)
        pre_trained_ans_vocab = get_embedding(pre_trained_vocab, ans_vocab) if opt.answer else None
        pre_trained_tgt_vocab = get_embedding(pre_trained_vocab, tgt_vocab)
        pre_trained_vocab = {'src':pre_trained_src_vocab, 'tgt':pre_trained_tgt_vocab}

    #========== save data ===========#
    data = {'settings': opt, 
            'dict': {'src': src_vocab,
                     'tgt': tgt_vocab,
                     'ans': ans_vocab if opt.answer else None,
                     'feature': feats_vocab, 
                     'pre-trained': pre_trained_vocab
            },
            'train': {'src': train_src_idx,
                      'tgt': train_tgt_idx,
                      'ans': train_ans_idx,
                      'feature': train_feat_idxs,
                      'copy':{'switch':train_copy_switch,
                              'tgt':train_copy_tgt}
            },
            'valid': {'src': valid_src_idx,
                      'tgt': valid_tgt_idx,
                      'ans': valid_ans_idx,
                      'feature': valid_feat_idxs,
                      'copy':{'switch':valid_copy_switch,
                              'tgt':valid_copy_tgt},
                      'tokens':{'src': valid_src_tokens,
                                'tgt': valid_tgt_tokens}
            }
        }
    
    return data, (train_final_indexes, valid_final_indexes), (train_src_indexes, valid_src_indexes)


def get_graph_data(raw, indexes):

    def get_edges(edges):
        edge_in = [[dep for dep in edge_set] for edge_set in edges]
        edge_out = [[dep for dep in edge_set] for edge_set in edges]
        for idx, edge_set in enumerate(zip(edge_in, edge_out)):
            set_in, set_out = edge_set[0], edge_set[1]
            for i, deps in enumerate(zip(set_in, set_out)):
                dep_in, dep_out = deps[0], deps[1]
                if dep_in not in ['self', 'similar', Constants.PAD_WORD] and not dep_in.startswith('re-'):
                    edge_in[idx][i] = Constants.PAD_WORD
                if dep_out not in ['self', 'similar', Constants.PAD_WORD] and dep_out.startswith('re-'):
                    edge_out[idx][i] = Constants.PAD_WORD
        return edge_in, edge_out

    dataset = {'indexes':[], 'type':[], 'is_ans': [], 'pos': [], 'is_tgt':[], 'edge_in':[], 'edge_out':[]}
    
    for index in tqdm(indexes, desc='   - (Loading graph data) -   '):
        sample = raw[index]
        nodes, edges = sample['nodes'], sample['edges']
        nodes_num = len(nodes)
        node_indexes, types, ans_tags, pos_tags, tgt_tags = [], [], [], [], []
        for idx, node in enumerate(nodes):
            node_indexes.append(node['index'])
            types.append(node['type'])
            ans_tags.append(node['ans'])
            pos_tags.append(node['pos'])
            tgt_tags.append(node['tag'])

            for i in range(nodes_num):
                if edges[idx][i] == '':
                    edges[idx][i] = Constants.PAD_WORD
                else:
                    edges[idx][i] = edges[idx][i].lower()
                    if edges[idx][i] not in ['similar', 'self'] and not edges[idx][i].startswith('re-'):
                        edges[i][idx] = 're-' + edges[idx][i]
        
        dataset['indexes'].append(node_indexes)
        dataset['type'].append(types)
        dataset['is_ans'].append(ans_tags)
        dataset['pos'].append(pos_tags)
        dataset['is_tgt'].append(tgt_tags)

        edge_in, edge_out = get_edges(edges)
        dataset['edge_in'].append(edge_in)
        dataset['edge_out'].append(edge_out)
    
    dataset['features'] = [dataset['type'], dataset['is_ans'], dataset['pos'], dataset['is_tgt']]
    
    return dataset


def process_bert_index(dicts, indexes):
    indexes = [dicts[idx] for idx in indexes]
    indexes = [i for idx in indexes for i in idx]

    return indexes


def graph_data(opt, final_indexes, bert_indexes=None):
    #========== get data ==========#
    train_file, valid_file = json_load(opt.train_graph), json_load(opt.valid_graph)
    train_data = get_graph_data(train_file, final_indexes[0])
    valid_data = get_graph_data(valid_file, final_indexes[1])
    
    #========== build vocabulary ==========#
    print('build node feature vocabularies')
    options = {'lower':False, 'mode':'size', 'tgt':False, 
               'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency}
    feats_vocab = [Vocab.from_opt(corpus=ft + fv, opt=options) for 
                   ft, fv in zip(train_data['features'], valid_data['features'])] if opt.node_feature else None
    print('build edge vocabularies')
    options = {'lower':True, 'mode':'size', 'tgt':False, 
               'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency}
    edge_in_corpus = [edge_set for edges in train_data['edge_in'] + valid_data['edge_in'] for edge_set in edges]
    edge_out_corpus = [edge_set for edges in train_data['edge_out'] + valid_data['edge_out'] for edge_set in edges]
    edge_in_vocab = Vocab.from_opt(corpus=edge_in_corpus, opt=options)
    edge_out_vocab = Vocab.from_opt(corpus=edge_out_corpus, opt=options)

    #========== word to index ==========#
    print(len(train_data['features']))
    train_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                        in zip(train_data['features'], feats_vocab)] if opt.node_feature else None
    valid_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                        in zip(valid_data['features'], feats_vocab)] if opt.node_feature else None
    train_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in train_data['edge_in']]
    valid_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in valid_data['edge_in']]
    train_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in train_data['edge_out']]
    valid_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in valid_data['edge_out']]
    
    #========== process indexes ===========#
    if opt.pretrained:
        train_indexes, valid_indexes = bert_indexes[0], bert_indexes[1]
        train_data['indexes'] = [[process_bert_index(dicts, node) for node in sent] for sent, dicts in zip(train_data['indexes'], train_indexes)]
        valid_data['indexes'] = [[process_bert_index(dicts, node) for node in sent] for sent, dicts in zip(valid_data['indexes'], valid_indexes)]

    #========== save data ===========#
    data = {'settings': opt, 
            'dict': {'feature': feats_vocab, 
                     'edge': {
                         'in': edge_in_vocab,
                         'out': edge_out_vocab
                     }
            },
            'train': {'index': train_data['indexes'],
                      'feature': train_feat_idxs,
                      'edge': {
                          'in': train_edge_in_idxs,
                          'out': train_edge_out_idxs
                      }
            },
            'valid': {'index': valid_data['indexes'],
                      'feature': valid_feat_idxs,
                      'edge': {
                          'in': valid_edge_in_idxs,
                          'out': valid_edge_out_idxs
                      }
            }
        }
    return data


def main(opt):
    print(opt.pretrained)
    sequences, final_indexes, bert_indexes = sequence_data(opt)
    graphs = graph_data(opt, final_indexes, bert_indexes)
    print("Saving data ......")
    torch.save(sequences, opt.save_sequence_data)
    torch.save(graphs, opt.save_graph_data)
    print('Saving Datasets ......')
    trainData = Dataset(sequences['train'], graphs['train'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=opt.copy)
    validData = Dataset(sequences['valid'], graphs['valid'], opt.batch_size, answer=opt.answer,
                        node_feature=opt.node_feature, copy=opt.copy)
    torch.save(trainData, opt.train_dataset)
    torch.save(validData, opt.valid_dataset)
    print("Done .")
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess.py')
    pargs.add_options(parser)
    opt = parser.parse_args()
    main(opt)
