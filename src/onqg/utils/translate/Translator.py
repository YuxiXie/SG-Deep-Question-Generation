import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from torch.autograd import Variable

from onqg.utils.translate.Beam import Beam
import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch

from nltk.translate import bleu_score


def add(tgt_list):
    tgt = []
    for b in tgt_list:
        tgt += b
    return tgt


def get_tokens(indexes, data):
    src, tgt = data['src'], data['tgt']
    srcs = [src[i] for i in indexes]
    golds = [[[w for w in tgt[i] if w not in [Constants.BOS_WORD, Constants.EOS_WORD]]] for i in indexes]
    return srcs, golds


class Translator(object):
    def __init__(self, opt, vocab, tokens, src_vocab):
        self.opt = opt
        self.max_token_seq_len = opt.max_token_tgt_len
        self.tokens = tokens
        if opt.gpus:
            cuda.set_device(opt.gpus[0])
        self.device = torch.device('cuda' if opt.gpus else 'cpu')
        self.vocab = vocab
    
    def translate_batch(self, model, inputs, max_length):
        
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq1 = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq2 = torch.stack(dec_partial_seq1).to(self.device)
            dec_partial_seq = dec_partial_seq2.view(-1, len_dec_seq)
            return dec_partial_seq
        
        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos
        
        def collect_active_inst_idx_list(inst_beams, pred_prob, copy_pred_prob, inst_idx_to_position_map, n_bm):
            active_inst_idx_list = []
            pred_prob = pred_prob.unsqueeze(0).view(len(inst_idx_to_position_map), n_bm, -1)
            copy_pred_prob = None if not self.opt.copy else copy_pred_prob.unsqueeze(0).view(len(inst_idx_to_position_map), n_bm, -1)
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                copy_prob = None if not self.opt.copy else copy_pred_prob[inst_position]
                is_inst_complete = inst_beams[inst_idx].advance(pred_prob[inst_position], copy_prob)
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm, layer=False):
            ''' Collect tensor parts associated to active instances. '''
            tmp_beamed_tensor = beamed_tensor[0] if layer else beamed_tensor
            _, *d_hs = tmp_beamed_tensor.size()

            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor if layer else [beamed_tensor]

            beamed_tensor = [layer_b_tensor.view(n_prev_active_inst, -1) for layer_b_tensor in beamed_tensor]
            beamed_tensor = [layer_b_tensor.index_select(0, curr_active_inst_idx) for layer_b_tensor in beamed_tensor]
            beamed_tensor = [layer_b_tensor.view(*new_shape) for layer_b_tensor in beamed_tensor]
            
            beamed_tensor = beamed_tensor if layer else beamed_tensor[0]
            
            return beamed_tensor

        with torch.no_grad():
            ### ========== Prepare data ========== ###
            if len(self.opt.gpus) > 1:
                model = model.module
            ### ========== Encode ========== ###
            seq_output, hidden = model.seq_encoder(inputs['seq-encoder'])
            inputs['encoder-transform']['seq_output'] = seq_output
            inputs['encoder-transform']['hidden'] = hidden
            node_input, hidden = model.encoder_transformer(inputs['encoder-transform'], max_length)
            inputs['graph-encoder']['nodes'] = node_input
            node_output, _ = model.graph_encoder(inputs['graph-encoder'])
            ##===== Decode =====##
            inputs['decoder-transform']['graph_output'] = node_output
            inputs['decoder-transform']['seq_output'] = seq_output
            inputs['decoder-transform']['hidden'] = hidden
            inputs['decoder']['enc_output'], inputs['decoder']['scores'], hidden = model.decoder_transformer(inputs['decoder-transform'])
            inputs['decoder']['hidden'] = hidden
            ### ========== Repeat for beam search ========== ###
            n_bm = self.opt.beam_size
            enc_output = inputs['decoder']['enc_output']
            if self.opt.layer_attn:
                n_inst, len_s, d_h = enc_output[0].size()
                inputs['decoder']['enc_output'] = [src_layer.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h) 
                                                    for src_layer in enc_output]
            else:
                n_inst, len_s, d_h = enc_output.size()
                inputs['decoder']['enc_output'] = enc_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            inputs['decoder']['src_seq'] = inputs['decoder']['src_seq'].repeat(1, n_bm).view(n_inst * n_bm, len_s)
            inputs['decoder']['ans_seq'] = inputs['decoder']['ans_seq'].repeat(1, n_bm).view(n_inst * n_bm, -1)
            inputs['decoder']['hidden'] = hidden.repeat(1, n_bm).view(n_inst * n_bm, -1)      # [h.repeat(1, n_bm).view(n_inst * n_bm, -1) for h in hidden]
            inputs['decoder']['feat_seqs'] = [feat_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s) 
                                                for feat_seq in inputs['decoder']['feat_seqs']] if self.opt.feature else None
            ### ========== Prepare beams ========== ###
            inst_dec_beams = [Beam(n_bm, self.vocab.size, self.opt.copy, device=self.device) for _ in range(n_inst)]
            ### ========== Bookkeeping for active or not ========== ###
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            ### ========== Decode ========== ###
            norm = nn.Softmax(dim=1)
            for len_dec_seq in range(1, self.max_token_seq_len + 1):
                n_active_inst = len(inst_idx_to_position_map)
                ### ===== decoder forward ===== ###
                inputs['decoder']['tgt_seq'] = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)     # (n_bm x batch_size) x len_dec_seq
                
                rst = model.decoder(inputs['decoder'])
                rst['pred'] = model.generator(rst['pred'])
                pred = rst['pred'][:, -1, :]
                pred = norm(pred)
                if self.opt.copy:
                    copy_pred, copy_gate = rst['copy_pred'][:, -1, :], rst['copy_gate'][:, -1, :]
                ### ===== log softmax ===== ###
                pred = norm(pred) + 1e-8
                pred_prob = pred
                copy_pred_log = None
                if self.opt.copy:
                    copy_gate = copy_gate.ge(0.5).type(torch.cuda.FloatTensor)
                    pred_prob_log = torch.log(pred_prob * ((1 - copy_gate).expand_as(pred_prob)) + 1e-25)
                    copy_pred_log = torch.log(copy_pred * (copy_gate.expand_as(copy_pred)) + 1e-25)
                else:
                    pred_prob_log = torch.log(pred_prob)
                ### ====== active list update ====== ###
                active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, pred_prob_log, copy_pred_log, 
                                                                    inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break   # all instances have finished their path to [EOS]
                ### ====== variables update ====== ###
                # Sentences which are still active are collected,
                # so the decoder will not run on completed sentences.
                n_prev_active_inst = len(inst_idx_to_position_map)
                active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
                active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

                inputs['decoder']['enc_output'] = collect_active_part(inputs['decoder']['enc_output'], active_inst_idx, n_prev_active_inst, 
                                                                      n_bm, layer=self.opt.layer_attn)
                inputs['decoder']['src_seq'] = collect_active_part(inputs['decoder']['src_seq'], active_inst_idx, n_prev_active_inst, n_bm)
                inputs['decoder']['ans_seq'] = collect_active_part(inputs['decoder']['ans_seq'], active_inst_idx, n_prev_active_inst, n_bm)
                inputs['decoder']['hidden'] = collect_active_part(inputs['decoder']['hidden'], active_inst_idx, n_prev_active_inst, n_bm)
                inputs['decoder']['feat_seqs'] = [collect_active_part(feat_seq, active_inst_idx, n_prev_active_inst, n_bm) 
                                                    for feat_seq in inputs['decoder']['feat_seqs']] if self.opt.feature else None
                
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        ### ========== Get hypothesis ========== ###
        all_hyp, all_scores = [], []
        all_copy_hyp, all_is_copy = [], []

        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores.append(scores[: self.opt.n_best])

            rsts = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:self.opt.n_best]]
            hyp = [rst['hyp'] for rst in rsts]
            all_hyp.append(hyp)
            if self.opt.copy:
                copy_hyp = [rst['cp_hyp'] for rst in rsts]
                is_copy = [rst['is_cp'] for rst in rsts]
                all_copy_hyp.append(copy_hyp)
                all_is_copy.append(is_copy)

        if self.opt.copy:
            return all_hyp, all_scores, all_is_copy, all_copy_hyp
        else:
            return all_hyp, all_scores
    
    def eval_batch(self, model, inputs, max_length, gold, copy_gold=None, copy_switch=None, batchIdx=None):
        
        def get_preds(seq, is_copy_seq=None, copy_seq=None, src_words=None, attn=None):
            pred = [idx for idx in seq if idx not in [Constants.PAD, Constants.EOS]]
            for i, _ in enumerate(pred):
                if self.opt.copy and is_copy_seq[i].item():
                    pred[i] = src_words[copy_seq[i].item() - self.vocab.size]
                else:
                    pred[i] = self.vocab.idxToLabel[pred[i]]
            return pred
        
        def src_tokens(src_words):
            if self.opt.pretrained.count('bert'):
                tmp_word, tmp_idx = '', 0
                for i, w in enumerate(src_words):
                    if not w.startswith('##'):
                        if tmp_word:
                            src_words[tmp_idx] = tmp_word
                            for j in range(tmp_idx + 1, i):
                                src_words[j] = ''
                            tmp_word, tmp_idx = w, i
                        else:
                            tmp_word += w.lstrip('##')
                src_words[tmp_idx] = tmp_word
                for j in range(tmp_idx + 1, i):
                    src_words[j] = ''
            raw_words = [word for word in src_words if word != '']
            return src_words, raw_words
            
        golds, preds, paras = [], [], []
        if self.opt.copy:
            all_hyp, _, all_is_copy, all_copy_hyp = self.translate_batch(model, inputs, max_length)
        else:
            all_hyp, _ = self.translate_batch(model, inputs, max_length)

        src_sents, golds = get_tokens(batchIdx, self.tokens)
        for i, seqs in tqdm(enumerate(all_hyp), mininterval=2, desc=' - (Translating)   ', leave=False):
            seq = seqs[0]
            src_words, raw_words = src_tokens(src_sents[i])
            if self.opt.copy:
                is_copy_seq, copy_seq = all_is_copy[i][0], all_copy_hyp[i][0]
                preds.append(get_preds(seq, is_copy_seq=is_copy_seq, copy_seq=copy_seq, src_words=src_words))
            else:
                preds.append(get_preds(seq))
            paras.append(raw_words)

        return {'gold':golds, 'pred':preds, 'para':paras}
    
    def eval_all(self, model, validData, output_sent=False):
        all_golds, all_preds = [], []
        if output_sent:
            all_paras = []

        for idx in tqdm(range(len(validData)), mininterval=2, desc='   - (Translating)   ', leave=False):
            ### ========== Prepare data ========== ###
            batch = validData[idx]
            inputs, max_lengths, gold, copy = preprocess_batch(batch, self.opt.edge_vocab_size, sparse=self.opt.sparse, feature=self.opt.feature,  
                                                               dec_feature=self.opt.dec_feature, copy=self.opt.copy, node_feature=self.opt.node_feature, 
                                                               device=self.device)
            copy_gold, copy_switch = copy[0], copy[1]
            ### ========== Translate ========== ###
            rst = self.eval_batch(model, inputs, max_lengths, gold, 
                                  copy_gold=copy_gold, copy_switch=copy_switch, 
                                  batchIdx=batch['raw-index'])
                
            all_golds += rst['gold']
            all_preds += rst['pred']
            if output_sent:
                all_paras += rst['para']
            
        
        bleu = bleu_score.corpus_bleu(all_golds, all_preds)

        if output_sent:
            return bleu, (all_golds, all_preds, all_paras)
        return bleu