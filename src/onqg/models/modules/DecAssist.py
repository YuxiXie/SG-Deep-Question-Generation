import torch
import torch.nn as nn


class DecInit(nn.Module):
    def __init__(self, d_enc, d_dec, n_enc_layer):
        self.d_enc_model = d_enc
        self.n_enc_layer = n_enc_layer
        self.d_dec_model = d_dec

        super(DecInit, self).__init__()

        self.initer = nn.Linear(self.d_enc_model * self.n_enc_layer, self.d_dec_model)
        self.tanh = nn.Tanh()
    
    def forward(self, hidden):
        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)
        return self.tanh(self.initer(hidden))


class StackedRNN(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout, rnn='lstm'):
        self.dropout = dropout
        self.num_layers = num_layers

        super(StackedRNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.name = rnn

        for _ in range(num_layers):
            if rnn == 'lstm':
                self.layers.append(nn.LSTMCell(input_size, rnn_size))
            elif rnn == 'gru':
                self.layers.append(nn.GRUCell(input_size, rnn_size))
            else:
                raise ValueError("Supported StackedRNN: LSTM, GRU")
            input_size = rnn_size
        
    def forward(self, inputs, hidden):
        if self.name == 'lstm':
            h_0, c_0 = hidden
        elif self.name == 'gru':
            h_0 = hidden
        h_1, c_1 = [], []
        
        for i, layer in enumerate(self.layers):
            if self.name == 'lstm':
                h_1_i, c_1_i = layer(inputs, (h_0[i], c_0[i]))
            elif self.name == 'gru':
                h_1_i = layer(inputs, h_0[i])
            inputs = h_1_i
            if i + 1 != self.num_layers:
                inputs = self.dropout(inputs)
            h_1.append(h_1_i)
            if self.name == 'lstm':
                c_1.append(c_1_i)
        
        h_1 = torch.stack(h_1)
        if self.name == 'lstm':
            c_1 = torch.stack(c_1)
            h_1 = (h_1, c_1)
        
        return inputs, h_1

