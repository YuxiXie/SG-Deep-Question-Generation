import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as funct

import onqg.dataset.Constants as Constants


class Loss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0
    
    def reset(self):
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        raise NotImplementedError
    
    def cuda(self):
        self.criterion.cuda()
    
    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate. ")
        self.acc_loss.backward()

class NLLLoss(Loss):

    _NAME = "NLLLoss"

    def __init__(self, opt, weight=None, mask=None, size_average=True, coverage_weight=0.1):
        
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask. ")
            weight[mask] = 0
        
        super(NLLLoss, self).__init__(self._NAME, nn.NLLLoss(weight=weight, size_average=size_average))

        try:
            self.opt = opt
            if opt.copy:
                self.copy_loss = nn.NLLLoss(size_average=False)
            self.coverage_weight = coverage_weight
        except:
            self.coverage_weight = coverage_weight
        
        self.KL = nn.KLDivLoss()
    
    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss
    
    def cal_loss(self, inputs):
        pred = inputs['pred']
        gold = inputs['gold']
        if self.opt.copy:
            copy_pred = inputs['copy_pred']
            copy_gold = inputs['copy_gold']
            copy_gate = inputs['copy_gate']
            copy_switch = inputs['copy_switch']
        if self.opt.coverage:
            coverage_pred = inputs['coverage_pred']
        
        batch_size = gold.size(0)
        gold = gold.contiguous()
        norm = nn.Softmax(dim=1)

        pred = pred.contiguous().view(-1, pred.size(2))
        pred = norm(pred)
        pred_prob_t = pred.contiguous().view(batch_size, -1, pred.size(1)) + 1e-8  # seq_len x batch_size x vocab_size

        if self.opt.copy:
            copy_pred_prob = copy_pred * copy_gate.expand_as(copy_pred) + 1e-8
            pred_prob = pred_prob_t * (1 - copy_gate).expand_as(pred_prob_t) + 1e-8

            copy_pred_prob_log = torch.log(copy_pred_prob)
            pred_prob_log = torch.log(pred_prob)
            copy_pred_prob_log = copy_pred_prob_log * (copy_switch.unsqueeze(2).expand_as(copy_pred_prob_log))
            pred_prob_log = pred_prob_log * ((1 - copy_switch).unsqueeze(2).expand_as(pred_prob_log))

            pred_prob_log = pred_prob_log.view(-1, pred_prob_log.size(2))
            copy_pred_prob_log = copy_pred_prob_log.view(-1, copy_pred_prob_log.size(2))
            
            pred_loss = self.criterion(pred_prob_log, gold.view(-1))
            copy_loss = self.copy_loss(copy_pred_prob_log, copy_gold.contiguous().view(-1))
            
            total_loss = pred_loss + copy_loss
        else:
            pred_prob_t_log = torch.log(pred_prob_t)
            pred_prob_t_log = pred_prob_t_log.view(-1, pred_prob_t_log.size(2))
            pred_loss = self.criterion(pred_prob_t_log, gold.view(-1))
            
            total_loss = pred_loss
        
        raw_loss = total_loss
        coverage_loss = None

        if self.opt.coverage:
            coverage_pred = [cv for cv in coverage_pred]

            coverage_loss = torch.sum(torch.stack(coverage_pred, 1), 1)
            coverage_loss = torch.sum(coverage_loss, 0)
            total_loss = total_loss + coverage_loss * self.coverage_weight

        return total_loss, coverage_loss, raw_loss

    def cal_loss_ner(self, pred, gold):
        device = gold.device
        golds = []
        for batch in gold:
            tmp_sent = torch.stack([w for w in batch if w.item() != Constants.PAD])
            golds.append(tmp_sent)
        golds = torch.cat(golds, dim=0).to(device)
        gold = golds.contiguous()
        
        pred = pred.contiguous().view(-1, pred.size(1))
        pred_prob_t_log = torch.log(pred + 1e-8)
        
        pred_loss = self.criterion(pred_prob_t_log, gold.view(-1))

        return pred_loss, gold
