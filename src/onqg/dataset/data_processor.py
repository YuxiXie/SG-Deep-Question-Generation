import torch
import onqg.dataset.Constants as Constants
from onqg.utils.mask import get_edge_mask


def preprocess_batch(batch, n_edge_type, sparse=True, feature=False, dec_feature=0, 
                     answer=False, ans_feature=False, copy=False, node_feature=False, 
                     device=None):
    """Get a batch by indexing to the Dataset object, then preprocess it to get inputs for the model
    Input: batch
        raw-index: idxBatch

        src: (wrap(srcBatch), lengths)
        tgt: wrap(tgtBatch)
        copy: (wrap(copySwitchBatch), wrap(copyTgtBatch))
        feat: (tuple(wrap(x) for x in featBatches), lengths)
        ans: (wrap(ansBatch), ansLengths)
        ans_feat: (tuple(wrap(x) for x in ansFeatBatches), ansLengths)

        edges: (wrap(edgeInBatch), wrap(edgeOutBatch), edge_lengths)
        edges_dict: (edgeInDict, edgeOutDict)
        tmp_nodes: (wrap(tmpNodesBatch), node_lengths)
        graph_index: (graphIndexBatch, node_index_length)
        graph_root: graphRootBatch
        node_feat: (tuple(wrap(x) for x in nodeFeatBatches), node_lengths)
    Output: 
        (1) inputs dict:
            seq-encoder: src_seq, lengths, feat_seqs
            graph-encoder: edges, mask, type, feat_seqs (, adjacent_matrix)
            encoder-transform: index, lengths, root, node_lengths
            decoder: tgt_seq, src_seq, feat_seqs
            decoder-transform: index
        (2) (max_node_num, max_node_size)
        (3) (generation, classification)
        (4) (copy_gold, copy_switch)
    """
    inputs = {'seq-encoder':{}, 'graph-encoder':{}, 'encoder-transform':{}, 
              'decoder':{}, 'decoder-transform':{}}
    ###===== RNN encoder =====###
    src_seq, tgt_seq = batch['src'], batch['tgt']
    src_seq, lengths = src_seq[0], src_seq[1]
    inputs['seq-encoder']['src_seq'], inputs['seq-encoder']['lengths'] = src_seq, lengths
    ###===== encoder transform =====###
    edges, nodes = batch['edges'][0], batch['tmp_nodes']
    nodes, node_lengths = nodes[0], nodes[1]
    graph_index = batch['graph_index']
    # graph_root = batch['graph_root']
    graph_index, index_lengths = graph_index[0], graph_index[1]
    inputs['encoder-transform']['index'], inputs['encoder-transform']['lengths'] = graph_index, index_lengths
    # inputs['encoder-transform']['root'] = graph_root
    inputs['encoder-transform']['node_lengths'] = node_lengths
    ###===== graph encoder =====###
    in_edge_mask, out_edge_mask = get_edge_mask(edges[0]), get_edge_mask(edges[1])
    if sparse:
        in_edge_mask = in_edge_mask.view(in_edge_mask.size(0), -1, max(node_lengths))
        out_edge_mask = out_edge_mask.view(in_edge_mask.size(0), -1, max(node_lengths))
        inputs['graph-encoder']['adjacent_matrix'] = batch['edges_dict']
        edge_type_list = torch.LongTensor([i for i in range(n_edge_type)]).to(device=device)
    inputs['graph-encoder']['edges'] = edge_type_list if sparse else (in_edge_mask, out_edge_mask)
    inputs['graph-encoder']['mask'] = (in_edge_mask, out_edge_mask)
    inputs['graph-encoder']['type'] = batch['node_feat'][0][0]
    ###===== classifier =====###
    classification = batch['node_feat'][0][-1]
    ###===== decoder transform =====###
    inputs['decoder-transform']['index'] = graph_index
    ###===== decoder =====###
    generation = tgt_seq[:, 1:]   # exclude [BOS] token
    inputs['decoder']['tgt_seq'] = tgt_seq[:, :-1]
    inputs['decoder']['src_seq'] = src_seq  # nodes
    inputs['decoder']['ans_seq'] = batch['ans'][0]
    ###===== auxiliary functions =====###
    src_feats, tgt_feats = None, None
    if feature:
        n_all_feature = len(batch['feat'][0])
        # split all features into src and tgt parts, src_feats are those embedded in the encoder
        src_feats = batch['feat'][0][:n_all_feature - dec_feature]
        if dec_feature:
            # dec_feature: the number of features embedded in the decoder
            tgt_feats = batch['feat'][0][n_all_feature - dec_feature:]
    inputs['seq-encoder']['feat_seqs'], inputs['decoder']['feat_seqs'] = src_feats, tgt_feats
   
    copy_gold, copy_switch = None, None
    if copy:
        copy_gold, copy_switch = batch['copy'][1], batch['copy'][0]
        copy_gold, copy_switch = copy_gold[:, 1:], copy_switch[:, 1:]

    node_feats = batch['node_feat'][0][1:-1] if node_feature else None
    inputs['graph-encoder']['feat_seqs'] = node_feats

    max_node_size = max(index_lengths)
    max_node_num = max(node_lengths)

    return inputs, (max_node_num, max_node_size), (generation, classification), (copy_gold, copy_switch)
