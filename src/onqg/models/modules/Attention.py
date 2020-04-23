import torch
import torch.nn as nn
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
                
        if mask is not None:
            attn = attn.masked_fill(mask, 0)
        sumattn = torch.sum(attn, dim=2, keepdim=True) + 1e-8
        attn = attn / sumattn
        
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim, is_coverage=False):
        super(ConcatAttention, self).__init__()

        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim

        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)

        self.sftmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.mask = None

        self.is_coverage = is_coverage
        if is_coverage:
            self.linear_cov = nn.Linear(1, att_dim, bias=False)
    
    def apply_mask(self, mask):
        self.mask = mask
    
    def forward(self, input, context, precompute=None, coverage=None, feat_inputs=None, feature=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        enc_output = torch.cat((context, feat_inputs), dim=2) if feature else context
        if precompute is None:
            precompute = self.linear_pre(enc_output)     # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp_sum = precompute + targetT.repeat(1, precompute.size(1), 1)  # batch x sourceL x att_dim

        if self.is_coverage:
            weighted_coverage = self.linear_cov(coverage.unsqueeze(2))  # batch x sourceL x att_dim
            tmp_sum += weighted_coverage

        tmp_activated = self.tanh(tmp_sum)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp_activated).view(tmp_sum.size(0), tmp_sum.size(1))  # batch x sourceL
        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        
        score = self.sftmax(energy)  # batch x sourceL
        
        weightedContext = torch.bmm(score.unsqueeze(1), context).squeeze(1)  # batch x dim

        if self.is_coverage:
            coverage = coverage + score  # batch x sourceL
            return weightedContext, score, precompute, coverage
        
        return weightedContext, score, precompute


class GatedSelfAttention(nn.Module):
    def __init__(self, dim, attn_dim=64, dropout=0.1):
        super(GatedSelfAttention, self).__init__()

        self.m_translate = nn.Linear(dim, attn_dim)     
        self.q_translate = nn.Linear(dim, attn_dim)  

        self.update = nn.Linear(2 * dim, dim, bias=False)

        self.gate = nn.Linear(2 * dim, dim, bias=False)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.has_dropout = True if dropout > 0 else False
    
    def forward(self, query, mask):
        raw = query

        memory = self.m_translate(query)  # b_sz x src_len x 64
        query = self.q_translate(query)

        energy = torch.bmm(query, memory.transpose(1, 2))      # b_sz x src_len x src_len
        energy = energy.masked_fill(mask, value=-1e12)

        score = torch.softmax(energy, dim=2)
        if self.has_dropout:
            score = self.dropout(score)
        context = torch.bmm(score, raw)

        inputs = torch.cat((raw, context), dim=2)

        f_t = torch.tanh(self.update(inputs))
        g_t = torch.sigmoid(self.gate(inputs))

        output = g_t * f_t + (1 - g_t) * raw

        return output, score


class GraphAttention(nn.Module):
    def __init__(self, d_q, d_v, alpha, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Linear(d_q + d_v, 1)
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, query, value, mask):
        """
        query - [batch_size, node_num * node_num, d_hidden]
        value - [batch_size, node_num * node_num, d_model]
        mask - [batch_size, node_num, node_num]
        """
        node_num = int(query.size(1) **0.5)
        query = query.view(-1, node_num, query.size(2))
        value = value.view(-1, node_num, value.size(2))     # (batch_size * node_num) x node_num x d_model

        pre_attention = torch.cat([query, value], dim=2)
        energy = self.leaky_relu(self.attention(pre_attention).squeeze(2))  # (batch_size * node_num) x node_num
        # energy = torch.ones((query.size(0) * node_num, node_num)).to(value.device)
        
        mask = mask.view(-1, node_num)
        zero_vec = -9e15 * torch.ones_like(energy)
        try:
            attention = torch.where(mask > 0, energy, zero_vec)
        except:
            print(mask.size(), zero_vec.size(), energy.size(), query.size(), value.size())
        
        scores = torch.softmax(attention, dim=1)    # (batch_size * node_num) x node_num
        scores = self.dropout(scores)

        value = torch.bmm(scores.unsqueeze(1), value).squeeze(1)   # (batch_size * node_num) x d_model
        value = value.view(-1, node_num, value.size(-1))

        return value
