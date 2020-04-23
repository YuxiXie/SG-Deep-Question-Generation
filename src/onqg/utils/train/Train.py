import os
import time
import math
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as funct

import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch


def record_log(logfile, step, loss, ppl, accu, bleu='unk', bad_cnt=0, lr='unk'):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(str(step) + ':\tloss=' + str(round(loss, 8)) + ',\tppl=' + str(round(ppl, 8)))
        f.write(',\tbleu=' + str(bleu) + ',\taccu=' + str(round(accu, 8)))
        f.write(',\tbad_cnt=' + str(bad_cnt) + ',\tlr=' + str(lr) + '\n')


class SupervisedTrainer(object):

    def __init__(self, model, loss, optimizer, translator, logger, opt, 
                 training_data, validation_data, src_vocab, graph_feature_vocab):
        self.model = model
        self.loss = loss
        self.class_loss = nn.BCELoss()
        if opt.gpus:
            self.class_loss.cuda()
        self.optimizer = optimizer
        self.translator = translator
        self.logger = logger
        self.opt = opt

        self.training_data = training_data
        self.validation_data = validation_data

        self.graph_feature_vocab = graph_feature_vocab

        self.cntBatch = 0
        self.best_ppl, self.best_bleu, self.best_accu, self.best_kl = math.exp(100), 0, 0, 100

    def cal_performance(self, loss_input):
        loss, cvl, rawl = self.loss.cal_loss(loss_input)

        gold, pred = loss_input['gold'], loss_input['pred']

        pred = pred.contiguous().view(-1, pred.size(2))
        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(Constants.PAD)

        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct, (cvl, rawl)
    
    def cal_class_performance(self, loss_input, device):
        pred, gold = loss_input['pred'], loss_input['gold']
        preds, golds = [], []
        for gbatch, pbatch in zip(gold, pred):
            for gw, pw in zip(gbatch, pbatch):
                if gw.item() != Constants.PAD:
                    golds.append(gw)
                    preds.append(pw)

        golds = torch.stack(golds, dim=0).to(device)
        golds = golds.eq(self.graph_feature_vocab[-1].labelToIdx[1]).float()    # TODO: magic number
        
        preds = torch.cat(preds, dim=0).to(device)
        golds, preds = golds.contiguous(), preds.contiguous()
        
        pred_loss = self.class_loss(preds, golds)

        preds = preds.ge(0.5).view(-1).float()

        correct = preds.eq(golds)
        correct = correct.sum().item()

        return pred_loss, correct
    
    def get_precision_and_recall(self, loss_input, device):
        pred, gold = loss_input['pred'], loss_input['gold']
        preds, golds = [], []
        for gbatch, pbatch in zip(gold, pred):
            for gw, pw in zip(gbatch, pbatch):
                if gw.item() != Constants.PAD:
                    golds.append(gw)
                    preds.append(pw)

        golds = torch.stack(golds, dim=0).to(device)
        golds = golds.eq(self.graph_feature_vocab[-1].labelToIdx[1]).float()    # TODO: magic number
        
        preds = torch.cat(preds, dim=0).to(device)
        golds, preds = golds.contiguous(), preds.contiguous()

        preds = preds.ge(0.5).view(-1).float()

        sum_p, sum_r = preds.eq(1).view(-1), golds.eq(1).view(-1)
        correct = preds.eq(golds).view(-1)
        
        cr_p, cr_r = (correct * sum_p).sum().item(), (correct * sum_r).sum().item()
        
        return [cr_p, sum_p.sum().item(), cr_r, sum_r.sum().item()]

    def save_model(self, better, eval_num):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': self.opt,
            'step': self.cntBatch}

        if self.opt.training_mode != 'classify':
            model_name = self.opt.save_model + '.chkpt'
            if better:
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')
            if eval_num != 'unk' and eval_num > self.best_bleu:
                self.best_bleu = eval_num
                model_name = self.opt.save_model + '_grt_' + str(round(eval_num * 100, 5)) + '_bleu4.chkpt'
                torch.save(checkpoint, model_name)
        elif self.opt.training_mode == 'classify' and better:
            model_name = self.opt.save_model + '_cls_' + str(round(eval_num * 100, 5)) + '_accuracy.chkpt'
            torch.save(checkpoint, model_name)
        elif better:
            model_name = self.opt.save_model + '_unf_' + str(round(eval_num, 5)) + '_KL.chkpt'
            torch.save(checkpoint, model_name)

    def eval_step(self, device, epoch):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()        

        total_loss = {'classify':0, 'generate':0, 'unify':0, 'coverage':0, 'nll':0}
        n_word_total, n_word_correct = 0, 0
        n_node_total, n_node_correct = 0, 0
        sample_num = 0

        precison, recall = [0, 0], [0, 0]

        with torch.no_grad():
            for idx in tqdm(range(len(self.validation_data)), mininterval=2, desc='  - (Validation) ', leave=False):
                batch = self.validation_data[idx]
                inputs, max_length, golds, copy = preprocess_batch(batch, self.opt.edge_vocab_size, sparse=self.opt.sparse, feature=self.opt.feature,  
                                                                   dec_feature=self.opt.dec_feature, copy=self.opt.copy, node_feature=self.opt.node_feature, 
                                                                   device=device)
                copy_gold, copy_switch = copy[0], copy[1]
                sample_num += len(golds[0])

                ### forward ###
                rst = self.model(inputs, max_length=max_length)

                loss_input = {'classification':{}, 'generation':{}, 'unify':{}}
                if self.opt.copy and self.opt.training_mode != 'classify':
                    loss_input['generation']['copy_pred'], loss_input['generation']['copy_gate'] = rst['generation']['copy_pred'], rst['generation']['copy_gate']
                    loss_input['generation']['copy_gold'], loss_input['generation']['copy_switch'] = copy_gold, copy_switch
                if self.opt.coverage and self.opt.training_mode != 'classify':
                    loss_input['generation']['coverage_pred'] = rst['generation']['coverage_pred']

                if self.opt.training_mode != 'generate':
                    loss_input['classification']['pred'] = rst['classification']
                    loss_input['classification']['gold'] = golds[1]
                    loss, n_correct_node = self.cal_class_performance(loss_input['classification'], device)
                    total_loss['classify'] += loss.item()
                    package = self.get_precision_and_recall(loss_input['classification'], device)
                    precison[0] += package[0]
                    recall[0] += package[2]
                    precison[1] += package[1]
                    recall[1] += package[3]
                if self.opt.training_mode != 'classify':
                    loss_input['generation']['pred'] = rst['generation']['pred']
                    loss_input['generation']['gold'] = golds[0]
                    loss, n_correct_word, loss_package = self.cal_performance(loss_input['generation'])
                    coverage_loss, nll_loss = loss_package[0], loss_package[1]
                    total_loss['generate'] += loss.item()
                    if self.opt.coverage:
                        total_loss['coverage'] += coverage_loss.item()
                    total_loss['nll'] += nll_loss.item()
                if self.opt.training_mode == 'unify':
                    loss_input['unify'] = rst['generation']['attention_scores']
                    kl_loss = funct.kl_div(torch.log(loss_input['unify'][0] + 1e-16), loss_input['unify'][1])
                    total_loss['unify'] += kl_loss.item()
                
                non_pad_mask = golds[0].ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_node = golds[1].ne(Constants.PAD).sum().item()
                if self.opt.training_mode != 'classify':
                    n_word_total += n_word
                    n_word_correct += n_correct_word
                if self.opt.training_mode != 'generate':
                    n_node_total += n_node
                    n_node_correct += n_correct_node

        outputs = {'classification':{}, 'generation':{}, 'unify':{}}
        if self.opt.training_mode != 'generate':
            outputs['classification']['loss'] = total_loss['classify'] / n_node_total
            outputs['classification']['correct'] = n_node_correct / n_node_total
            print('\n***', precison[0] / precison[1], recall[0] / recall[1], '***\n')
        if self.opt.training_mode != 'classify':
            outputs['generation']['loss'] = total_loss['generate'] / n_word_total
            outputs['generation']['correct'] = n_word_correct / n_word_total
            outputs['generation']['bleu'] = 'unk'
            outputs['generation']['perplexity'] = math.exp(min(total_loss['nll'] / n_word_total, 16))
            if self.opt.coverage:
                outputs['generation']['coverage'] = total_loss['coverage'] / sample_num
            if outputs['generation']['perplexity'] <= self.opt.translate_ppl or outputs['generation']['perplexity'] > self.best_ppl:
                if self.cntBatch % self.opt.translate_steps == 0: 
                    outputs['generation']['bleu'] = self.translator.eval_all(self.model, self.validation_data)   
        if self.opt.training_mode == 'unify':
            outputs['unify'] = total_loss['unify']

        return outputs

    def train_epoch(self, device, epoch):
        ''' Epoch operation in training phase'''
        if self.opt.extra_shuffle and epoch > self.opt.curriculum:
            self.logger.info('Shuffling...')
            self.training_data.shuffle()

        self.model.train()

        total_loss = {'classify':0, 'generate':0, 'unify':0, 'coverage':0, 'nll':0}
        n_word_total, n_word_correct = 0, 0
        n_node_total, n_node_correct = 0, 0
        report_total_loss = {'classify':0, 'generate':0, 'unify':0, 'coverage':0, 'nll':0}
        report_n_word_total, report_n_word_correct = 0, 0
        report_n_node_total, report_n_node_correct = 0, 0
        sample_num = 0

        batch_order = torch.randperm(len(self.training_data))

        for idx in tqdm(range(len(self.training_data)), mininterval=2, desc='  - (Training)   ', leave=False):

            batch_idx = batch_order[idx] if epoch > self.opt.curriculum else idx
            batch = self.training_data[batch_idx]

            ##### ==================== prepare data ==================== #####
            inputs, max_length, golds, copy = preprocess_batch(batch, self.opt.edge_vocab_size, sparse=self.opt.sparse, feature=self.opt.feature,  
                                                               dec_feature=self.opt.dec_feature, copy=self.opt.copy, node_feature=self.opt.node_feature, 
                                                               device=device)
            copy_gold, copy_switch = copy[0], copy[1]
            sample_num += len(golds[0])
                
            ##### ==================== forward ==================== #####
            self.model.zero_grad()
            self.optimizer.zero_grad()
            
            rst = self.model(inputs, max_length=max_length)

            ##### ==================== backward ==================== #####
            loss_input = {'classification':{}, 'generation':{}, 'unify':{}}

            if self.opt.copy and self.opt.training_mode != 'classify':
                loss_input['generation']['copy_pred'], loss_input['generation']['copy_gate'] = rst['generation']['copy_pred'], rst['generation']['copy_gate']
                loss_input['generation']['copy_gold'], loss_input['generation']['copy_switch'] = copy_gold, copy_switch
            if self.opt.coverage and self.opt.training_mode != 'classify':
                loss_input['generation']['coverage_pred'] = rst['generation']['coverage_pred']

            if self.opt.training_mode != 'generate':
                loss_input['classification']['pred'] = rst['classification']
                loss_input['classification']['gold'] = golds[1]
                loss, n_correct_node = self.cal_class_performance(loss_input['classification'], device)
                cls_loss = loss
                total_loss['classify'] += loss.item()
                report_total_loss['classify'] += loss.item()
            if self.opt.training_mode != 'classify':
                loss_input['generation']['pred'] = rst['generation']['pred']
                loss_input['generation']['gold'] = golds[0]
                loss, n_correct_word, loss_package = self.cal_performance(loss_input['generation'])
                gen_loss = loss
                coverage_loss, nll_loss = loss_package[0], loss_package[1]
                total_loss['generate'] += loss.item()
                report_total_loss['generate'] += loss.item()
                if self.opt.coverage:
                    total_loss['coverage'] += coverage_loss.item()
                    report_total_loss['coverage'] += coverage_loss.item()
                total_loss['nll'] += nll_loss.item()
                report_total_loss['nll'] += nll_loss.item()
            if self.opt.training_mode == 'unify':
                loss_input['unify'] = rst['generation']['attention_scores']
                kl_loss = funct.kl_div(torch.log(loss_input['unify'][0] + 1e-16), loss_input['unify'][1])
                total_loss['unify'] += kl_loss.item()
                report_total_loss['unify'] += kl_loss.item()
            
            self.cntBatch += 1
            if self.opt.training_mode == 'unify':
                # if (self.cntBatch // 16) % 512 == 0 and self.cntBatch > 10000:     # TODO: fix this magic number
                #     loss = kl_loss
                ratio = self.cntBatch // 8000 + 4     # TODO: fix this magic number
                if (self.cntBatch // 128) % ratio == 0:     # TODO: fix this magic number
                    loss = cls_loss
                else:
                    loss = gen_loss
            
            if len(self.opt.gpus) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if math.isnan(loss.item()) or loss.item() > 1e20:
                print('catch NaN')
                import ipdb; ipdb.set_trace()

            self.optimizer.backward(loss)
            self.optimizer.step()

            ##### ==================== record epoch report & step report ==================== #####
            non_pad_mask = golds[0].ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()

            if self.opt.training_mode != 'classify':
                n_word_total += n_word
                n_word_correct += n_correct_word
                report_n_word_total += n_word
                report_n_word_correct += n_correct_word
            if self.opt.training_mode != 'generate':
                n_node = golds[1].ne(Constants.PAD).sum().item()
                n_node_total += n_node
                n_node_correct += n_correct_node
                report_n_node_total += n_node
                report_n_node_correct += n_correct_node

            ##### ==================== evaluation ==================== #####
            if self.cntBatch % self.opt.valid_steps == 0:                
                ### ========== evaluation on dev ========== ###
                valid_results = self.eval_step(device, epoch)

                better = False
                valid_eval = 0

                if self.opt.training_mode != 'generate':
                    report_avg_loss = report_total_loss['classify'] / report_n_node_total
                    report_avg_accu = report_n_node_correct / report_n_node_total * 100
                    report_total_loss['classify'], report_n_node_total, report_n_node_correct = 0, 0, 0
                    better = valid_results['classification']['correct'] > self.best_accu
                    if better:
                        self.best_accu = valid_results['classification']['correct']
                    valid_eval = valid_results['classification']['correct']
                    self.logger.info('  +  Training accuracy: {accu:3.3f} %, loss: {loss:3.5f}'.format(accu=report_avg_accu, loss=report_avg_loss))
                    self.logger.info('  +  Validation accuracy: {accu:3.3f} %, loss: {loss:3.5f}'.format(accu=valid_results['classification']['correct'] * 100, 
                                                                                                      loss=valid_results['classification']['loss']))
                if self.opt.training_mode != 'classify':
                    report_avg_loss = report_total_loss['generate'] / report_n_word_total
                    report_avg_ppl = math.exp(min(report_total_loss['nll'] / report_n_word_total, 16))
                    report_avg_accu = report_n_word_correct / report_n_word_total
                    if self.opt.coverage:
                        report_avg_coverage = report_total_loss['coverage'] / sample_num
                        report_total_loss['coverage'] = 0
                        self.logger.info('  +  Training coverage loss: {loss:2.5f}'.format(loss=report_avg_coverage))
                        self.logger.info('  +  Validation coverage loss: {loss:2.5f}'.format(loss=valid_results['generation']['coverage']))
                    report_total_loss['generate'], report_total_loss['nll'] = 0, 0
                    report_n_word_correct, report_n_word_total = 0, 0
                    better = valid_results['generation']['perplexity'] < self.best_ppl
                    if better:
                        self.best_ppl = valid_results['generation']['perplexity']
                    valid_eval = valid_results['generation']['bleu']
                if self.opt.training_mode == 'unify':
                    report_avg_kldiv = report_total_loss['unify'] / sample_num
                    report_total_loss['unify'] = 0
                    # better = valid_results['unify'] < self.best_kl
                    # if better:
                    #     self.best_kl = valid_results['unify']
                    # valid_eval = valid_results['unify']
                    self.logger.info('  +  Training kl-div loss: {loss:2.5f}'.format(loss=report_avg_kldiv))
                    self.logger.info('  +  Validation kl-div loss: {loss:2.5f}'.format(loss=valid_results['unify']))
                sample_num = 0
                
                ### ========== update learning rate ========== ###
                self.optimizer.update_learning_rate(better)

                if self.opt.training_mode != 'classify':
                    record_log(self.opt.logfile_train, step=self.cntBatch, loss=report_avg_loss, ppl=report_avg_ppl, 
                               accu=report_avg_accu, bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)
                    record_log(self.opt.logfile_dev, step=self.cntBatch, loss=valid_results['generation']['loss'], 
                               ppl=valid_results['generation']['perplexity'], accu=valid_results['generation']['correct'],
                               bleu=valid_results['generation']['bleu'], bad_cnt=self.optimizer._bad_cnt, 
                               lr=self.optimizer._learning_rate)

                if self.opt.save_model:
                    self.save_model(better, valid_eval)

                self.model.train()

        if self.opt.training_mode == 'generate':
            loss_per_word = total_loss['generate'] / n_word_total
            perplexity = math.exp(min(loss_per_word, 16))
            accuracy = n_word_correct / n_word_total * 100
            outputs = (perplexity, accuracy)
        elif self.opt.training_mode == 'classify':
            outputs = n_node_correct / n_node_total * 100
        else:
            outputs = total_loss['unify']

        return outputs

    def train(self, device):
        ''' Start training '''
        self.logger.info(self.model)

        for epoch_i in range(self.opt.epoch):
            self.logger.info('')
            self.logger.info(' *  [ Epoch {0} ]:   '.format(epoch_i))
            start = time.time()
            results = self.train_epoch(device, epoch_i + 1)

            if self.opt.training_mode == 'generate':
                self.logger.info(' *  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(ppl=results[0], accu=results[1]))
            elif self.opt.training_mode == 'classify':
                self.logger.info(' *  - (Training)   accuracy: {accu: 3.3f} %'.format(accu=results))
            else:
                self.logger.info(' *  - (Training)   loss: {loss: 2.5f}'.format(loss=results))
            print('                ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch_i))
        