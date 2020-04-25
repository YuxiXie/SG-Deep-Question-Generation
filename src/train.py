import os
import xargs
import argparse

import math
import time
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import cuda

import onqg.dataset.Constants as Constants
from onqg.dataset.Dataset import Dataset

from onqg.utils.model_builder import build_model
from onqg.utils.train.Loss import NLLLoss
from onqg.utils.train.Optim import Optimizer
from onqg.utils.train import SupervisedTrainer
from onqg.utils.translate import Translator


def main(opt, logger):
    logger.info('My PID is {0}'.format(os.getpid()))
    logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
    logger.info(opt)

    if torch.cuda.is_available() and not opt.gpus:
        logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
    if opt.gpus:
        if opt.cuda_seed > 0:
            torch.cuda.manual_seed(opt.cuda_seed)
        cuda.set_device(opt.gpus[0])
    logger.info('My seed is {0}'.format(torch.initial_seed()))
    logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

    ###### ==================== Loading Options ==================== ######
    if opt.checkpoint:
        checkpoint = torch.load(opt.checkpoint)

    ###### ==================== Loading Dataset ==================== ######
    opt.sparse = True if opt.sparse else False
    # logger.info('Loading sequential data ......')
    # sequences = torch.load(opt.sequence_data)
    # seq_vocabularies = sequences['dict']
    # logger.info('Loading structural data ......')
    # graphs = torch.load(opt.graph_data)
    # graph_vocabularies = graphs['dict']

    ### ===== load pre-trained vocabulary ===== ###
    logger.info('Loading sequential data ......')
    sequences = torch.load(opt.sequence_data)
    seq_vocabularies = sequences['dict']
    logger.info('Loading pre-trained vocabulary ......')
    if opt.pre_trained_vocab:
        if not opt.pretrained:
            opt.pre_trained_src_emb = seq_vocabularies['pre-trained']['src']
        opt.pre_trained_tgt_emb = seq_vocabularies['pre-trained']['tgt']
        if opt.answer:
            opt.pre_trained_ans_emb = seq_vocabularies['pre-trained']['src']
    
    ### ===== wrap datasets ===== ###
    logger.info('Loading Dataset objects ......')
    trainData = torch.load(opt.train_dataset)
    validData = torch.load(opt.valid_dataset)
    trainData.batchSize = validData.batchSize = opt.batch_size
    trainData.numBatches = math.ceil(len(trainData.src) / trainData.batchSize)
    validData.numBatches = math.ceil(len(validData.src) / validData.batchSize)
    
    logger.info('Preparing vocabularies ......')
    opt.src_vocab_size = seq_vocabularies['src'].size
    opt.tgt_vocab_size = seq_vocabularies['tgt'].size
    opt.feat_vocab = [fv.size for fv in seq_vocabularies['feature']] if opt.feature else None

    logger.info('Loading structural data ......')
    graphs = torch.load(opt.graph_data)
    graph_vocabularies = graphs['dict']
    del graphs

    opt.edge_vocab_size = graph_vocabularies['edge']['in'].size
    opt.node_feat_vocab = [fv.size for fv in graph_vocabularies['feature'][1:-1]] if opt.node_feature else None
    
    logger.info(' * vocabulary size. source = %d; target = %d' % (opt.src_vocab_size, opt.tgt_vocab_size))
    logger.info(' * number of training batches. %d' % len(trainData))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    ##### =================== Prepare Model =================== #####
    device = torch.device('cuda' if opt.gpus else 'cpu')
    trainData.device = validData.device = device
    checkpoint = checkpoint if opt.checkpoint else None
    
    model, parameters_cnt = build_model(opt, device, checkpoint=checkpoint)
    del checkpoint

    logger.info(' * Number of parameters to learn = %d' % parameters_cnt)

    ##### ==================== Prepare Optimizer ==================== #####
    optimizer = Optimizer.from_opt(model, opt)

    ##### ==================== Prepare Loss ==================== #####
    weight = torch.ones(opt.tgt_vocab_size)
    weight[Constants.PAD] = 0
    loss = NLLLoss(opt, weight, size_average=False)
    if opt.gpus:
        loss.cuda()

    ##### ==================== Prepare Translator ==================== #####
    translator = Translator(opt, seq_vocabularies['tgt'], sequences['valid']['tokens'], seq_vocabularies['src'])
    
    ##### ==================== Training ==================== #####
    trainer = SupervisedTrainer(model, loss, optimizer, translator, logger, 
                                opt, trainData, validData, seq_vocabularies['src'],
                                graph_vocabularies['feature'])
    del model
    del trainData
    del validData
    del seq_vocabularies['src']
    del graph_vocabularies['feature']
    trainer.train(device)


if __name__ == '__main__':
    ##### ==================== parse the options ==================== #####
    parser = argparse.ArgumentParser(description='train.py')
    xargs.add_data_options(parser)
    xargs.add_model_options(parser)
    xargs.add_train_options(parser)
    opt = parser.parse_args()

    ##### ==================== prepare the logger ==================== #####
    logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
    if opt.log_home:
        log_file_name = os.path.join(opt.log_home, log_file_name)
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    main(opt, logger)
