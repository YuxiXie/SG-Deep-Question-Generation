import numpy as np
import torch

import onqg.dataset.Constants as Constants


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), 
                                 diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_slf_attn_mask(attn_mask, lengths, device=None):
    ''' For masking out according to the given attention matrix '''
    max_length = torch.max(lengths, 0)[0].item()
    mask = torch.ones((lengths.size(0), max_length, max_length), device=device, dtype=torch.uint8)

    for idx, sample in enumerate(attn_mask):
        seq_len = int(len(sample) **0.5)
        sample = sample.view(seq_len, seq_len)
        pad_sample = sample if max_length == seq_len else torch.cat((sample, torch.ones((max_length - seq_len, seq_len), 
                                                                    dtype=torch.uint8)), dim=0)
        mask[idx].narrow(1, 0, seq_len).copy_(pad_sample)
    mask = mask.view(-1, max_length, max_length)
    
    return mask


def get_slf_window_mask(seq, window_size=3, separate=-1):
    ''' For masking out the words in distance:
        only allow a word to attend to those near to it
        'near' means: within window_size words
    '''
    assert window_size >= 0, "Window size cannot be smaller than zero! "

    sz_b, len_s = seq.size()

    slf_window_mask = torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8)

    if separate >= 0:
        tmp_seq = [[w.item() for w in sent] for sent in seq]
        indexes = [sent.index(separate) for sent in tmp_seq]
    else:
        for idx in range(len_s):
            for i in range(idx - window_size, idx + window_size + 1):
                if i >= 0 and i < len_s:
                    slf_window_mask[idx][i] = 0

    slf_window_mask = slf_window_mask.unsqueeze(0).repeat(sz_b, 1, 1)  # b x ls x ls

    if separate >= 0:
        for b_idx in range(sz_b):
            sep = indexes[b_idx]
            for idx in range(len_s):
                sep_final = tmp_seq[b_idx].index(separate, sep + 1)
                if idx == 0:
                    for i in range(0, sep_final + 1):
                        slf_window_mask[b_idx][idx][i] = 0
                elif idx == sep:
                    for i in range(0, sep + 1):
                        slf_window_mask[b_idx][idx][i] = 0
                elif idx == sep_final:
                    slf_window_mask[b_idx][idx][0] = 0
                    for i in range(sep + 1, sep_final + 1):
                        slf_window_mask[b_idx][idx][i] = 0
                else:
                    slf_window_mask[b_idx][idx][0] = 0
                    for i in range(idx - window_size, idx + window_size + 1):
                        if i >= 0 and i < len_s:
                            if (idx <= sep and i <= sep) or (idx > sep and i > sep):
                                slf_window_mask[b_idx][idx][i] = 0
                    if idx <= sep:
                        slf_window_mask[b_idx][idx][sep] = 0
                    else:                    
                        slf_window_mask[b_idx][idx][sep_final] = 0

    return slf_window_mask


def get_edge_mask(edges):
    ''' Get mask matrix for edges
    edges - [batch_size, node_num * node_num]
    return - [batch_size, node_num, node_num]
    '''
    len_edges = edges.size(1)
    node_num = int(len_edges **0.5)
    
    mask = edges.eq(Constants.PAD)
    mask = mask.view(-1, node_num, node_num)

    return mask