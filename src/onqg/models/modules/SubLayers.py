''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onqg.models.modules.Attention import ScaledDotProductAttention
from onqg.models.modules.MaxOut import MaxOut


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, addition_input=0, dropout=0.1, attn_dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model + addition_input, n_head * d_k)
        self.w_vs = nn.Linear(d_model + addition_input, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + addition_input + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + addition_input + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=attn_dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is None:
            mask = None
        else:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.onelayer = d_hid == d_in
        if self.onelayer:   #   just to reduce the number of parameters
            self.w = nn.Linear(d_in, d_in, bias=False)
            self.tanh = nn.Tanh()
        else:
            self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
            self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        
        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        if self.onelayer:
            output = self.w(x)
            output = self.tanh(output)
        else:
            output = x.transpose(1, 2)
            output = self.w_2(F.relu(self.w_1(output)))
            output = output.transpose(1, 2)  # batch_size x seq_length x d_word_vec

        output = self.dropout(output)        
        output = self.layer_norm(output + residual)
        
        return output


class Propagator(nn.Module):
    def __init__(self, state_dim, dropout=0.1):
        super(Propagator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )
    
    def forward(self, cur_state, in_vec, out_vec):
        """
        cur_state - [batch_size, node_num, d_model]
        in_vec - [batch_size, node_num, d_model]
        out_vec - [batch_size, node_num, d_model]
        """
        a = torch.cat([in_vec, out_vec, cur_state], dim=2)
        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat([in_vec, out_vec, r * cur_state], dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * cur_state + z * h_hat    # batch_size x node_num x d_model
        return output