''' Define the Layers '''
import torch
import torch.nn as nn
from onqg.models.modules.SubLayers import MultiHeadAttention, PositionwiseFeedForward, Propagator
from onqg.models.modules.Attention import GatedSelfAttention, GraphAttention
import onqg.dataset.Constants as Constants


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, slf_attn, d_inner, n_head, d_k, d_v, 
                 dropout=0.1, attn_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = slf_attn
        if slf_attn == 'multi-head':
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, attn_dropout=attn_dropout)
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        else:
            self.gated_slf_attn = GatedSelfAttention(d_model, d_k, dropout=attn_dropout)

    def forward(self, enc_input, src_seq, non_pad_mask=None, slf_attn_mask=None, layer_id=-1):
        if self.slf_attn == 'gated':
            mask = (src_seq == Constants.PAD).unsqueeze(2) if slf_attn_mask is None else slf_attn_mask
            enc_output, enc_slf_attn = self.gated_slf_attn(enc_input, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask

            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class GraphEncoderLayer(nn.Module):
    '''GGNN & GAT Layer'''
    def __init__(self, d_hidden, d_model, alpha, feature=False, dropout=0.1, attn_dropout=0.1):
        super(GraphEncoderLayer, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.feature = feature

        self.edge_num = 3   # TODO: fix this magic number
        bias_list = [False, False, False]
        self.edge_in_list = nn.ModuleList([nn.Linear(d_hidden, d_model, bias=bias_list[i]) for i in range(self.edge_num)])
        self.edge_out_list = nn.ModuleList([nn.Linear(d_hidden, d_model, bias=bias_list[i]) for i in range(self.edge_num)])
        # self.edge_in_emb = nn.Linear(d_hidden, d_model)
        # self.edge_out_emb = nn.Linear(d_hidden, d_model)

        self.graph_in_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.graph_out_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.output_gate = Propagator(d_model, dropout=dropout)

    def forward(self, nodes, mask, node_type, feat_hidden=None):
        ###=== concatenation ===###
        node_hidden = nodes     # batch_size x node_num x d_model
        # if self.feature:
        #     node_hidden = torch.cat((node_hidden, feat_hidden), dim=-1)   # batch_size x node_num x d_hidden
        ###=== transform using edge matrix ===###
        in_masks = [(node_type == tag).float().unsqueeze(2).repeat(1, 1, self.d_model).to(nodes.device) 
                    for tag in range(2, 2 + self.edge_num)]
        node_in_hidden = torch.sum(torch.stack([in_emb(node_hidden) * in_masks[idx] 
                            for idx, in_emb in enumerate(self.edge_in_list)], dim=0), dim=0)
        out_masks = [(node_type == tag).float().unsqueeze(2).repeat(1, 1, self.d_model).to(nodes.device) 
                    for tag in range(2, 2 + self.edge_num)]
        node_out_hidden = torch.sum(torch.stack([out_emb(node_hidden) * out_masks[idx] 
                            for idx, out_emb in enumerate(self.edge_out_list)], dim=0), dim=0)     # batch_size x node_num x d_model
        # node_in_hidden = self.edge_in_emb(node_hidden)
        # node_out_hidden = self.edge_out_emb(node_hidden)
        ###=== graph attention ===###
        node_hidden = node_hidden.unsqueeze(2).repeat(1, 1, nodes.size(1), 1).view(nodes.size(0), -1, self.d_hidden)
        node_in_hidden = self.graph_in_attention(node_hidden, node_in_hidden.repeat(1, nodes.size(1), 1), mask[0])
        node_out_hidden = self.graph_out_attention(node_hidden, node_out_hidden.repeat(1, nodes.size(1), 1), mask[1])
        ###=== gated recurrent unit ===###
        node_output = self.output_gate(nodes, node_in_hidden, node_out_hidden)

        return node_output


class SparseGraphEncoderLayer(nn.Module):
    '''Sparse GGNN & GAT Layer'''
    def __init__(self, d_hidden, d_model, alpha, edge_bias=False, feature=False, dropout=0.1, attn_dropout=0.1):
        super(SparseGraphEncoderLayer, self).__init__()
        self.edge_bias = edge_bias
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.feature = feature

        self.graph_in_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.graph_out_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.output_gate = Propagator(d_model, dropout=dropout)

    def forward(self, nodes, edges, mask, adjacent_matrixes, feat_hidden=None, 
                edge_hidden_bias=None):
        ###=== concatenation ===###
        node_hidden = nodes
        if self.feature:
            node_hidden = torch.cat((node_hidden, feat_hidden), dim=-1)   # batch_size x node_num x d_hidden
        ###=== forward ===###
        node_in_hidden = torch.zeros((nodes.size(0), nodes.size(1), nodes.size(1), self.d_model)).to(nodes.device)
        node_out_hidden = torch.zeros((nodes.size(0), nodes.size(1), nodes.size(1), self.d_model)).to(nodes.device)
        in_matrixes, out_matrixes = adjacent_matrixes[0], adjacent_matrixes[1]

        for sample_id, data in enumerate(zip(node_hidden, in_matrixes, out_matrixes)):
            sample, in_matrix, out_matrix = data[0], data[1], data[2]
            # in/out_matrix - [real_node_num, real_neighbor_num]      
            for idx, indexes in enumerate(zip(in_matrix, out_matrix)):
                in_index, out_index = indexes[0], indexes[1]
                # in/out_index - [real_neighbor_num]
                for wrap in in_index:
                    vector = torch.matmul(sample[wrap[0]], edges[0][wrap[1]])   # [d_model]
                    node_in_hidden[sample_id][idx].narrow(0, wrap[0], 1).copy_(vector)
                for wrap in out_index:
                    vector = torch.matmul(sample[wrap[0]], edges[1][wrap[1]])   # [d_model]
                    node_out_hidden[sample_id][idx].narrow(0, wrap[0], 1).copy_(vector)

        ###=== graph-self-attention ===###
        node_in_hidden = node_in_hidden.view(nodes.size(0), -1, self.d_model)
        node_out_hidden = node_out_hidden.view(nodes.size(0), -1, self.d_model)
        node_hidden = node_hidden.unsqueeze(2).repeat(1, 1, nodes.size(1), 1).view(nodes.size(0), -1, self.d_hidden)
        node_in_hidden = self.graph_in_attention(node_hidden, node_in_hidden, mask[0])
        node_out_hidden = self.graph_out_attention(node_hidden, node_out_hidden, mask[1])
        ###=== gated recurrent unit ===###
        node_output = self.output_gate(nodes, node_in_hidden, node_out_hidden)

        return node_output
