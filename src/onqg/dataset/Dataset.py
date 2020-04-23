from __future__ import division

import math
import random
import numpy as np

import torch
from torch import cuda

import onqg.dataset.Constants as Constants


class Dataset(object):

    def __init__(self, seq_datasets, graph_datasets, batchSize, 
                 copy=False, answer=False, ans_feature=False, 
                 feature=False, node_feature=False, opt_cuda=False):
        self.src, self.tgt = seq_datasets['src'], seq_datasets['tgt']
        self.has_tgt = True if self.tgt else False

        self.graph_index = graph_datasets['index']
        # self.graph_root = graph_datasets['root']
        self.edge_in = graph_datasets['edge']['in']
        self.edge_out = graph_datasets['edge']['out']

        self.answer = answer
        self.ans = seq_datasets['ans'] if answer else None
        self.ans_feature_num = len(seq_datasets['ans_feature']) if ans_feature else 0
        self.ans_features = seq_datasets['ans_feature'] if self.ans_feature_num else None

        self.feature_num = len(seq_datasets['feature']) if feature else 0
        self.features = seq_datasets['feature'] if self.feature_num else None

        self.node_feature_num = len(graph_datasets['feature']) if node_feature else 0
        self.node_features = graph_datasets['feature'] if self.node_feature_num else None

        self.copy = copy
        self.copy_switch = seq_datasets['copy']['switch'] if copy else None
        self.copy_tgt = seq_datasets['copy']['tgt'] if copy else None
        
        self._update_data()
        
        if opt_cuda:
            cuda.set_device(opt_cuda[0])
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)
    
    def _update_data(self):
        """sort all data by lengths of source text"""
        self.idxs = list(range(len(self.src)))
        lengths = [s.size(0) for s in self.src]
        RAW = [lengths, self.src, self.idxs]

        DATA = list(zip(*RAW))
        DATA.sort(key=lambda x:x[0])

        self.src = [d[1] for d in DATA]
        self.idxs = [d[2] for d in DATA]

        if self.tgt:
            self.tgt = [self.tgt[idx] for idx in self.idxs]
        if self.copy:
            self.copy_switch = [self.copy_switch[idx] for idx in self.idxs]
            self.copy_tgt = [self.copy_tgt[idx] for idx in self.idxs]
        if self.feature_num:
            self.features = [[feature[idx] for idx in self.idxs] for feature in self.features]
        if self.answer:
            self.ans = [self.ans[idx] for idx in self.idxs]
            if self.ans_feature_num:
                self.ans_features = [[feature[idx] for idx in self.idxs] for feature in self.ans_features]
        
        self.edge_in_dict = self._get_edge_dict(self.edge_in)
        self.edge_out_dict = self._get_edge_dict(self.edge_out)

    def _get_edge_dict(self, edges):
        edges_dict = []
        for sample in edges:
            edge_dict = []
            for edge_list in sample:
                edge_dict.append([(idx, edge.item()) for idx, edge in enumerate(edge_list) if edge.item() != Constants.PAD])
            edges_dict.append(edge_dict)
        return edges_dict

    def _batchify(self, data, align_right=False, include_lengths=False, src_len=None):
        """get data in a batch while applying padding, return length if needed"""
        if src_len:
            lengths = src_len
        else:
            lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out
    
    def _graph_length_info(self, indexes, edges):
        indexes = [index for sample in indexes for index in sample]
        index_length = [len(index) for index in indexes]

        node_length = [len(edge) for edge in edges]
        nodes = [torch.Tensor([i for i in range(length)]) for length in node_length]
        tmpNodesBatch = self._batchify(nodes, src_len=node_length)
        return index_length, node_length, tmpNodesBatch
    
    def _pad_edges(self, edges, node_length, align_right=False):
        max_length = max(node_length)
        edge_lengths = []
        
        outs = []
        for edge, data_length in zip(edges, node_length):
            out = edge[0].new_full((max_length, max_length), Constants.PAD)
            for i, e in enumerate(edge):
                offset = max_length - data_length if align_right else 0
                out[i].narrow(0, offset, data_length).copy_(e)
            outs.append(out.view(-1))
            edge_lengths.append(data_length * data_length)
        outs = torch.stack(outs, dim=0)     # batch_size x (max_length * max_length)

        return outs, edge_lengths

    def __getitem__(self, index):
        """get the exact batch using index, and transform data into Tensor form"""
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        srcBatch, lengths = self._batchify(self.src[index * self.batchSize: (index + 1) * self.batchSize],
                                           align_right=False, include_lengths=True)
        tgtBatch = None
        if self.tgt:
            tgtBatch = self._batchify(self.tgt[index * self.batchSize: (index + 1) * self.batchSize])

        idxBatch = self.idxs[index * self.batchSize: (index + 1) * self.batchSize]

        graphIndexBatch = [self.graph_index[i] for i in idxBatch]
        # graphRootBatch = [self.graph_root[i] for i in idxBatch]
        edgeInBatch, edgeOutBatch = [self.edge_in[i] for i in idxBatch], [self.edge_out[i] for i in idxBatch]
        edgeInDict, edgeOutDict = [self.edge_in_dict[i] for i in idxBatch], [self.edge_out_dict[i] for i in idxBatch]
        node_index_length, node_lengths, tmpNodesBatch = self._graph_length_info(graphIndexBatch, edgeInBatch)
        edgeInBatch, edge_lengths = self._pad_edges(edgeInBatch, node_lengths)
        edgeOutBatch, _ = self._pad_edges(edgeOutBatch, node_lengths)

        nodeFeatBatches = None
        if self.node_feature_num:
            nodeFeatBatches = [
                self._batchify([feat[i] for i in idxBatch], src_len=node_lengths) for feat in self.node_features
            ]
        
        featBatches = None
        if self.feature_num:
            featBatches = [
                self._batchify(feat[index * self.batchSize: (index + 1) * self.batchSize], 
                               src_len=lengths) for feat in self.features
            ]
        
        copySwitchBatch, copyTgtBatch = None, None
        if self.copy:
            copySwitchBatch = self._batchify(self.copy_switch[index * self.batchSize: (index + 1) * self.batchSize])
            copyTgtBatch = self._batchify(self.copy_tgt[index * self.batchSize: (index + 1) * self.batchSize])
        
        ansBatch, ansFeatBatches = None, None
        if self.answer:
            ansBatch, ansLengths = self._batchify(self.ans[index * self.batchSize: (index + 1) * self.batchSize],
                                                  align_right=False, include_lengths=True)
            if self.ans_feature_num:
                ansFeatBatches = [
                    self._batchify(feat[index * self.batchSize: (index + 1) * self.batchSize], src_len=ansLengths) 
                    for feat in self.ans_features
                ]

        def wrap(b):
            if b is None:
                return b
            b = torch.stack([x for x in b], dim=0).contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1).to(self.device)
        edge_lengths = torch.LongTensor(edge_lengths).view(1, -1).to(self.device)
        indices = range(len(srcBatch))
        
        rst = {}
        rst['indice'] = indices
        rst['src'] = (wrap(srcBatch), lengths)
        rst['raw-index'] = idxBatch
        if self.has_tgt:
            rst['tgt'] = wrap(tgtBatch)
        if self.copy:
            rst['copy'] = (wrap(copySwitchBatch), wrap(copyTgtBatch))
        if self.answer:
            ansLengths = torch.LongTensor(ansLengths).view(1, -1).to(self.device)
            rst['ans'] = (wrap(ansBatch), ansLengths)
            if self.ans_feature_num:
                rst['ans_feat'] = (tuple(wrap(x) for x in ansFeatBatches), ansLengths)
        if self.feature_num:
            rst['feat'] = (tuple(wrap(x) for x in featBatches), lengths)
        
        rst['edges'] = ((wrap(edgeInBatch), wrap(edgeOutBatch)), edge_lengths)
        rst['edges_dict'] = (edgeInDict, edgeOutDict)
        rst['tmp_nodes'] = (wrap(tmpNodesBatch), node_lengths)
        rst['graph_index'] = (graphIndexBatch, node_index_length)
        # rst['graph_root'] = graphRootBatch
        if self.node_feature_num:
            rst['node_feat'] = (tuple(wrap(x) for x in nodeFeatBatches), node_lengths)

        return rst

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        """shuffle the order of data in every batch"""

        def shuffle_group(start, end, NEW):
            """shuffle the order of samples with index from start to end"""
            RAW = [self.src[start:end], self.tgt[start:end], self.idxs[start:end]]
            DATA = list(zip(*RAW))
            index = torch.randperm(len(DATA))

            src, tgt, idx = zip(*[DATA[i] for i in index])
            NEW['SRCs'] += list(src)
            NEW['TGTs'] += list(tgt)
            NEW['IDXs'] += list(idx)

            if self.answer:
                ans = [self.ans[start:end][i] for i in index]
                NEW['ANSs'] += ans
                if self.ans_feature_num:
                    ansft = [[feature[start:end][i] for i in index] for feature in self.ans_features]
                    for i in range(self.ans_feature_num):
                        NEW['ANSFTs'][i] += ansft[i]
            
            if self.feature_num:
                ft = [[feature[start:end][i] for i in index] for feature in self.features]
                for i in range(self.feature_num):
                    NEW['FTs'][i] += ft[i]

            if self.copy:
                cpswt = [self.copy_switch[start:end][i] for i in index]
                cptgt = [self.copy_tgt[start:end][i] for i in index]
                NEW['COPYSWTs'] += cpswt
                NEW['COPYTGTs'] += cptgt 
            
            return NEW

        assert self.tgt != None, "shuffle is only aimed for training data (with target given)"
        
        NEW = {'SRCs':[], 'TGTs':[], 'IDXs':[]}
        if self.copy:
            NEW['COPYSWTs'], NEW['COPYTGTs'] = [], []
        if self.feature_num:
            NEW['FTs'] = [[] for i in range(self.feature_num)]
        if self.answer:
            NEW['ANSs'] = []
            if self.ans_feature_num:
                NEW['ANSFTs'] = [[] for i in range(self.ans_feature_num)]

        shuffle_all = random.random()
        if shuffle_all > 0.75:      # fix this magic number later
            start, end = 0, self.batchSize * self.numBatches
            NEW = shuffle_group(start, end, NEW)
        else:
            for batch_idx in range(self.numBatches):
                start = batch_idx * self.batchSize
                end = start + self.batchSize

                NEW = shuffle_group(start, end, NEW)
            
        self.src, self.tgt, self.idxs = NEW['SRCs'], NEW['TGTs'], NEW['IDXs']
        if self.copy:
            self.copy_switch, self.copy_tgt = NEW['COPYSWTs'], NEW['COPYTGTs']
        if self.answer:
            self.ans = NEW['ANSs'] 
            if self.ans_feature_num:
                self.ans_features = NEW['ANSFTs']
        if self.feature_num:
            self.features = NEW['FTs']
