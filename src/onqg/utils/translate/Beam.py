import time
import numpy as np

import torch
import onqg.dataset.Constants as Constants


class Beam():
    def __init__(self, size, vocab_size, copy=False, device=None):
        self.vocab_size = vocab_size
        self.size = size
        self.copy = copy
        self.device = device

        self._done = False
        # Scores for each translation on the beam
        self.scores = torch.zeros((size, ), dtype=torch.float, device=device)
        self.all_scores = []
        self.all_length = []
        # Backpointers at each time step
        self.prev_ks = []
        # Outputs at each time step
        self.next_ys = [torch.full((size, ), Constants.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = Constants.BOS
        self.next_ys_cp = [torch.full((size, ), Constants.PAD, dtype=torch.long, device=device)]
        self.next_ys_cp[0][0] = Constants.BOS
        # Attentions (matrix) for each time step
        self.attn = []
        # Whether copy for each time step
        self.is_copy = []
    
    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()
        #return self.next_ys[-1]
    
    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]
    
    @property
    def done(self):
        return self._done

    def advance(self, pred_prob, copy_pred_prob=None, attn=None):
        "Update beam status and check if finished or not."
        num_words = pred_prob.size(1)
        raw_num_words = num_words
        if self.copy:
            assert copy_pred_prob is not None
            num_src_words = copy_pred_prob.size(1)
            num_words += num_src_words
            pred_prob = torch.cat((pred_prob, copy_pred_prob), dim=1)
        
        # Accumulate length for those who hasn't finished yet
        if len(self.prev_ks) > 0:
            finish_index = self.next_ys[-1].eq(Constants.EOS)   # get the EOS indexes
            if any(finish_index):
                pred_prob.masked_fill_(finish_index.unsqueeze(1).expand_as(pred_prob), -float('inf'))
                for idx in range(self.size):
                    if self.next_ys[-1][idx] == Constants.EOS:
                        pred_prob[idx][Constants.EOS] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for idx in range(self.size):
                cur_length[idx] += 0 if self.next_ys[-1][idx] == Constants.EOS else 1            

        # Sum the previous scores
        if len(self.prev_ks) > 0:
            prev_score = self.all_scores[-1]
            now_acc_score = pred_prob + prev_score.unsqueeze(1).expand_as(pred_prob)
            beam_lk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(torch.FloatTensor(self.size).fill_(1).to(self.device))
            beam_lk = pred_prob[0]
        
        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        # best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort
        # self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        predict = best_scores_id - prev_k * num_words
        if self.copy:
            is_copy = predict.ge(torch.LongTensor(self.size).fill_(raw_num_words).to(self.device)).long()
        else:
            is_copy = 0
        final_predict = predict * (1 - is_copy) + is_copy * Constants.UNK

        if len(self.prev_ks) > 0:
            self.all_length.append(cur_length.index_select(0, prev_k))  # 
            self.all_scores.append(now_acc_score.view(-1).index_select(0, best_scores_id))
        else:
            self.all_scores.append(self.scores)

        self.prev_ks.append(prev_k)
        self.next_ys.append(final_predict)
        self.next_ys_cp.append(predict)
        self.is_copy.append(is_copy)
        if attn:
            self.attn.append(attn.index_select(0, prev_k))

        # End condition is when top-of-beam is EOS.
        if all(self.next_ys[-1].eq(Constants.EOS)):
            self._done = True

        return self._done
    
    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)
    
    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]
    
    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp, copy_hyp = [], []
        if len(self.attn) > 0:
            attn = []
        if self.copy:
            is_copy = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k].item())
            if len(self.attn) > 0:
                attn.append(self.attn[j][k])
            if self.copy:
                is_copy.append(self.is_copy[j][k])
            copy_hyp.append(self.next_ys_cp[j + 1][k])
            k = self.prev_ks[j][k]
        
        rst = {'hyp':hyp[::-1], 'cp_hyp':copy_hyp[::-1]}
        if len(self.attn) > 0:
            rst['attn'] = torch.stack(attn[::-1])
        if self.copy:
            rst['is_cp'] = is_copy[::-1]

        return rst

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."  
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k)['hyp'] for k in keys]
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq
    
    