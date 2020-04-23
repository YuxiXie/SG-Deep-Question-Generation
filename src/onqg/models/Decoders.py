import torch
import torch.nn as nn
from torch.autograd import Variable

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import ConcatAttention
from onqg.models.modules.MaxOut import MaxOut
from onqg.models.modules.DecAssist import StackedRNN, DecInit


class RNNDecoder(nn.Module):
    """
    Input: (1) inputs['tgt_seq']
           (2) inputs['src_seq']
           (3) inputs['src_indexes']
           (4) inputs['enc_output']
           (5) inputs['hidden']
           (6) inputs['feat_seqs']
    Output: (1) rst['pred']
            (2) rst['attn']
            (3) rst['context']
            (4) rst['copy_pred']; rst['copy_gate']
            (5) rst['coverage_pred']

    """
    def __init__(self, n_vocab, ans_n_vocab, d_word_vec, d_model, n_layer, n_rnn_enc_layer, 
                 rnn, d_k, feat_vocab, d_feat_vec, d_rnn_enc_model, d_enc_model, 
                 n_enc_layer, input_feed, copy, answer, coverage, layer_attn, 
                 maxout_pool_size, dropout, device=None):
        self.name = 'rnn'

        super(RNNDecoder, self).__init__()

        self.n_layer = n_layer
        self.layer_attn = layer_attn
        self.coverage = coverage
        self.copy = copy
        self.maxout_pool_size = maxout_pool_size
        input_size = d_word_vec

        self.input_feed = input_feed
        if input_feed:
            input_size += d_rnn_enc_model + d_enc_model

        self.ans_emb = nn.Embedding(ans_n_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.answer = answer
        tmp_in = d_word_vec if answer else d_rnn_enc_model
        self.decInit = DecInit(d_enc=tmp_in, d_dec=d_model, n_enc_layer=n_rnn_enc_layer)

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
        feat_size = len(feat_vocab) * d_feat_vec if self.feature else 0

        self.d_enc_model = d_rnn_enc_model + d_enc_model

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.rnn = StackedRNN(n_layer, input_size, d_model, dropout, rnn=rnn)
        self.attn = ConcatAttention(self.d_enc_model + feat_size, d_model, d_k, coverage)

        self.readout = nn.Linear((d_word_vec + d_model + self.d_enc_model), d_model)
        self.maxout = MaxOut(maxout_pool_size)

        if copy:
            self.copy_switch = nn.Linear(self.d_enc_model + d_model, 1)
        
        self.hidden_size = d_model
        self.dropout = nn.Dropout(dropout)
        self.device = device

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['ans_n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'], opt['n_rnn_enc_layer'],
                   opt['rnn'], opt['d_k'], opt['feat_vocab'], opt['d_feat_vec'], opt['d_rnn_enc_model'], 
                   opt['d_enc_model'], opt['n_enc_layer'], opt['input_feed'], opt['copy'], opt['answer'],
                   opt['coverage'], opt['layer_attn'], opt['maxout_pool_size'], opt['dropout'], 
                   opt['device'])

    def attn_init(self, context):
        if isinstance(context, list):
            context = context[-1]
        if isinstance(context, tuple):
            context = torch.cat(context, dim=-1)
        batch_size = context.size(0)
        hidden_sizes = (batch_size, self.d_enc_model)
        return Variable(context.data.new(*hidden_sizes).zero_(), requires_grad=False)

    def forward(self, inputs, max_length=300):
        tgt_seq, src_seq, ans_seq = inputs['tgt_seq'], inputs['src_seq'], inputs['ans_seq']
        enc_output, hidden = inputs['enc_output'], inputs['hidden']
        feat_seqs = inputs['feat_seqs']

        src_pad_mask = Variable(src_seq.data.eq(Constants.PAD).float(), requires_grad=False, volatile=False)
        if self.layer_attn:
            n_enc_layer = len(enc_output)
            src_pad_mask = src_pad_mask.repeat(1, n_enc_layer)
            enc_output = torch.cat(enc_output, dim=1)
        
        feat_inputs = None
        if self.feature:
            feat_inputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, n_enc_layer, 1)

        dec_outputs, coverage_output, copy_output, copy_gate_output = [], [], [], []
        cur_context = self.attn_init(enc_output)

        if self.answer:
            ans_words = torch.sum(self.ans_emb(ans_seq), dim=1)
            hidden = self.decInit(ans_words).unsqueeze(0)
        else:
            hidden = self.decInit(hidden).unsqueeze(0)
        # ans_words = torch.sum(self.ans_emb(ans_seq), dim=1)
        # hidden = self.decInit(hidden).unsqueeze(0)
        tmp_context, tmp_coverage = None, None

        dec_input = self.word_emb(tgt_seq)      
        
        self.attn.apply_mask(src_pad_mask)

        attention_scores = None
        tag = False

        dec_input = dec_input.transpose(0, 1)
        for seq_idx, dec_input_emb in enumerate(dec_input.split(1)):
            dec_input_emb = dec_input_emb.squeeze(0)
            raw_dec_input_emb = dec_input_emb
            if self.input_feed:
                dec_input_emb = torch.cat((dec_input_emb, cur_context), dim=1)
            dec_output, hidden = self.rnn(dec_input_emb, hidden)

            if self.coverage:
                if tmp_coverage is None:
                    tmp_coverage = Variable(torch.zeros((enc_output.size(0), enc_output.size(1))))
                    if self.device:
                        tmp_coverage = tmp_coverage.to(self.device)
                cur_context, attn, tmp_context, next_coverage = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                                          coverage=tmp_coverage, feat_inputs=feat_inputs,
                                                                          feature=self.feature)
                avg_tmp_coverage = tmp_coverage / max(1, seq_idx)
                coverage_loss = torch.sum(torch.min(attn, avg_tmp_coverage), dim=1)
                tmp_coverage = next_coverage
                coverage_output.append(coverage_loss)
            else:
                cur_context, attn, tmp_context = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                           feat_inputs=feat_inputs, feature=self.feature)
            
            attention_scores = attn if not tag else attn + attention_scores
            tag = True
            
            if self.copy:
                copy_prob = self.copy_switch(torch.cat((dec_output, cur_context), dim=1))
                copy_prob = torch.sigmoid(copy_prob)

                if self.layer_attn:
                    attn = attn.view(attn.size(0), n_enc_layer, -1)
                    attn = attn.sum(1)
                
                copy_output.append(attn)
                copy_gate_output.append(copy_prob)
            
            readout = self.readout(torch.cat((raw_dec_input_emb, dec_output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            
            dec_outputs.append(output)
        
        dec_output = torch.stack(dec_outputs).transpose(0, 1)

        sum_attention_scores = torch.sum(attention_scores, dim=1, keepdim=True) + 1e-8
        attention_scores = attention_scores / sum_attention_scores

        rst = {}
        rst['pred'], rst['attn'], rst['context'] = dec_output, attn, cur_context
        rst['attention_scores'] = (attention_scores, inputs['scores'])
        if self.copy:
            copy_output = torch.stack(copy_output).transpose(0, 1)
            copy_gate_output = torch.stack(copy_gate_output).transpose(0, 1)
            rst['copy_pred'], rst['copy_gate'] = copy_output, copy_gate_output
        if self.coverage:
            coverage_output = torch.stack(coverage_output).transpose(0, 1)
            rst['coverage_pred'] = coverage_output
        return rst


class DecoderTransformer(nn.Module):
    '''
        seq_output - [batch_size, seq_length, dim_seq_enc]
        graph_output - [batch_size, node_num, dim_graph_enc]
        indexes_list - [batch_size, node_num, index_num] (list)
    '''
    def __init__(self, layer_attn, device=None):
        super(DecoderTransformer, self).__init__()
        self.layer_attn = layer_attn
        self.device = device
    
    def forward(self, inputs):
        seq_output, hidden = inputs['seq_output'], inputs['hidden']
        graph_output, indexes_list = inputs['graph_output'], inputs['index']

        batch_size, seq_length = seq_output.size(0), seq_output.size(1)
        dim_graph_enc = graph_output.size(-1) if not self.layer_attn else graph_output[-1].size(-1)
        if 'scores' in inputs:
            scores = inputs['scores']
            distribution = torch.full((batch_size, seq_length), 1e-8).to(self.device)
        
        if self.layer_attn:
            graph_hidden_states = [torch.full((batch_size, seq_length, dim_graph_enc), 1e-8).to(self.device) for _ in range(len(graph_output))]
        else:
            graph_hidden_states = torch.full((batch_size, seq_length, dim_graph_enc), 1e-8).to(self.device)
        
        graph_node_sizes = torch.full((batch_size, seq_length), 0).to(self.device)
        
        for sample_idx, indexes in enumerate(indexes_list):
            ##=== for each sample ===##
            for node_idx, index in enumerate(indexes):
                ##=== for each node ===##
                for i in index:
                    ##=== for each word ===##
                    if self.layer_attn:
                        for idx in range(len(graph_hidden_states)):
                            graph_hidden_states[idx][sample_idx].narrow(0, i, 1).add_(graph_output[idx][sample_idx][node_idx])
                    else:
                        graph_hidden_states[sample_idx].narrow(0, i, 1).add_(graph_output[sample_idx][node_idx])
                    graph_node_sizes[sample_idx][i] += 1
                    if 'scores' in inputs:
                        distribution[sample_idx][i] = scores[sample_idx][node_idx]
        
        for i in range(batch_size):
            for j in range(seq_length):
                if graph_node_sizes[i][j].item() < 1:
                    graph_node_sizes[i][j] = 1
        
        if self.layer_attn:
            graph_hidden_states = [x / graph_node_sizes.unsqueeze(2).repeat(1, 1, dim_graph_enc) for x in graph_hidden_states]
        else:
            graph_hidden_states = graph_hidden_states / graph_node_sizes.unsqueeze(2).repeat(1, 1, dim_graph_enc)
        if 'scores' in inputs:
            distribution = distribution / graph_node_sizes

        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)

        distribution = distribution if 'scores' in inputs else None
        if self.layer_attn:
            enc_output = [torch.cat((graph_output, seq_output), dim=-1) for graph_output in graph_hidden_states]
        else:
            enc_output = torch.cat((graph_hidden_states, seq_output), dim=-1)

        return enc_output, distribution, hidden