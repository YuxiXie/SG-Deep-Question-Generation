import math
import torch.nn as nn

from onqg.models.Models import UnifiedModel
from onqg.models.Encoders import RNNEncoder, GraphEncoder, EncoderTransformer, SparseGraphEncoder, TransfEncoder
from onqg.models.Decoders import RNNDecoder, DecoderTransformer


def build_encoder(opt, answer=False, graph=False):
    if graph:
        options = {'n_edge_type':opt.edge_vocab_size, 'd_model':opt.d_graph_enc_model, 
                   'd_rnn_enc_model':opt.d_seq_enc_model, 'n_layer':opt.n_graph_enc_layer,
                   'alpha':opt.alpha, 'd_feat_vec':opt.d_feat_vec, 'feat_vocab':opt.node_feat_vocab,
                   'layer_attn':opt.layer_attn, 'dropout':opt.dropout, 'attn_dropout':opt.attn_dropout}
        model = SparseGraphEncoder.from_opt(options) if opt.sparse else GraphEncoder.from_opt(options)
    else:
        if opt.pretrained and not answer:
            options = {'pretrained':opt.pretrained, 'n_vocab':opt.src_vocab_size, 'layer_attn':opt.layer_attn}
            
            model = TransfEncoder.from_opt(options)
            for para in model.parameters():
                para.requires_grad = False
            
            return model
        
        feat_vocab = opt.feat_vocab
        if feat_vocab:
            n_all_feat = len(feat_vocab)
            feat_vocab = feat_vocab[:n_all_feat - opt.dec_feature]

        options = {'n_vocab':opt.src_vocab_size, 'd_word_vec':opt.d_word_vec, 'd_model':opt.d_seq_enc_model,
                   'n_layer':opt.n_seq_enc_layer, 'brnn':opt.brnn, 'rnn':opt.enc_rnn, 'slf_attn':opt.slf_attn, 
                   'feat_vocab':feat_vocab, 'd_feat_vec':opt.d_feat_vec, 'dropout':opt.dropout}
        
        model = RNNEncoder.from_opt(options)

    return model       


def build_decoder(opt, device):
    if opt.dec_feature:
        n_all_feat = len(opt.feat_vocab)
        feat_vocab = opt.feat_vocab[n_all_feat - opt.dec_feature:]
    else:
        feat_vocab = None
    
    options = {'n_vocab':opt.tgt_vocab_size, 'ans_n_vocab':opt.src_vocab_size, 'd_word_vec':opt.d_word_vec, 'd_model':opt.d_dec_model,
               'n_layer':opt.n_dec_layer, 'n_rnn_enc_layer':opt.n_seq_enc_layer, 'rnn':opt.dec_rnn, 'd_k':opt.d_k, 
               'feat_vocab':feat_vocab, 'd_feat_vec':opt.d_feat_vec, 'd_enc_model':opt.d_graph_enc_model, 
               'd_rnn_enc_model':opt.d_seq_enc_model, 'n_enc_layer':opt.n_graph_enc_layer, 'input_feed':opt.input_feed, 
               'copy':opt.copy, 'coverage':opt.coverage, 'layer_attn':opt.layer_attn, 'answer':opt.answer,
               'maxout_pool_size':opt.maxout_pool_size, 'dropout':opt.dropout, 'device':device}
    model = RNNDecoder.from_opt(options)
    
    return model


def initialize(model, opt):
    parameters_cnt = 0
    for name, para in model.named_parameters():
        size = list(para.size())
        local_cnt = 1
        for d in size:
            local_cnt *= d

        if not opt.pretrained or not name.count('seq_encoder'):
            if para.dim() == 1:
                para.data.normal_(0, math.sqrt(6 / (1 + para.size(0))))
            else:
                nn.init.xavier_normal(para, math.sqrt(3))

            parameters_cnt += local_cnt
    
    if opt.pre_trained_vocab:
        assert opt.d_word_vec == 300, "Dimension of word vectors must equal to that of pretrained word-embedding"
        if not opt.pretrained:
            model.seq_encoder.word_emb.weight.data.copy_(opt.pre_trained_src_emb)
        model.decoder.word_emb.weight.data.copy_(opt.pre_trained_tgt_emb)
        if opt.answer:
            model.decoder.ans_emb.weight.data.copy_(opt.pre_trained_ans_emb)
    
    if opt.proj_share_weight:
        weight = model.decoder.maxout(model.decoder.word_emb.weight.data)
        model.generator.weight.data.copy_(weight)

    return model, parameters_cnt


def build_model(opt, device, separate=-1, checkpoint=None):
    ## build model ##
    seq_encoder = build_encoder(opt)
    encoder_transformer = EncoderTransformer(opt.d_seq_enc_model, d_k=opt.d_k, device=device)
    graph_encoder = build_encoder(opt, graph=True)
    if opt.d_seq_enc_model != opt.d_graph_enc_model:
        graph_encoder.activate = nn.Sequential(
            nn.Linear(opt.d_seq_enc_model, opt.d_graph_enc_model, bias=False),
            nn.Tanh()
        )
    else:
        graph_encoder.activate = nn.Tanh()
    decoder_transformer = DecoderTransformer(opt.layer_attn, device=device)
    decoder = build_decoder(opt, device)

    model = UnifiedModel(opt.training_mode, seq_encoder, graph_encoder, encoder_transformer,
                         decoder, decoder_transformer)
    
    model.generator = nn.Linear(opt.d_dec_model // opt.maxout_pool_size, opt.tgt_vocab_size, bias=False)
    model.classifier = nn.Sequential(
        nn.Linear(opt.d_graph_enc_model, 1, bias=False),
        nn.Sigmoid()
    )
    
    model, parameters_cnt = initialize(model, opt)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    
    ## move to gpus ##
    model = model.to(device)
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus)
    
    return model, parameters_cnt
