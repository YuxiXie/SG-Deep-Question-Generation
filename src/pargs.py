import argparse


def add_options(parser):
    ##### ========== Data Files ========== #####
    parser.add_argument('-train_src', help="Path to the training source data")
    parser.add_argument('-train_tgt', help="Path to the training target data")
    parser.add_argument('-valid_src', help="Path to the validation source data")
    parser.add_argument('-valid_tgt', help="Path to the validation target data")
    parser.add_argument('-train_dataset', help="Path to the training dataset object")
    parser.add_argument('-valid_dataset', help="Path to the validation dataset object")

    parser.add_argument('-train_graph', help="Path to the training source graph data")
    parser.add_argument('-valid_graph', help="Path to the validation source graph data")

    parser.add_argument('-train_ans', default='', help="Path to the training answer")
    parser.add_argument('-valid_ans', default='', help="Path to the validation answer")

    parser.add_argument('-feature', default=False, action='store_true')
    parser.add_argument('-node_feature', default=False, action='store_true')
    parser.add_argument('-train_feats', default=[], nargs='+', type=str, help="Train files of source features")
    parser.add_argument('-valid_feats', default=[], nargs='+', type=str, help="Valid files of source features")
    
    parser.add_argument('-answer', default=False, action='store_true')
    parser.add_argument('-ans_feature', default=False, action='store_true')
    parser.add_argument('-train_ans_feats', default=[], nargs='+', type=str, help="Train files of answer features")
    parser.add_argument('-valid_ans_feats', default=[], nargs='+', type=str, help="Valid files of answer features")

    ##### ========== Data Preprocess Options ========== #####
    parser.add_argument('-copy', default=False, action='store_true')
    
    parser.add_argument('-src_seq_length', type=int, default=300)
    parser.add_argument('-tgt_seq_length', type=int, default=100)

    parser.add_argument('-src_vocab_size', type=int, default=50000)
    parser.add_argument('-tgt_vocab_size', type=int, default=50000)  
    parser.add_argument('-src_words_min_frequency', type=int, default=1)
    parser.add_argument('-tgt_words_min_frequency', type=int, default=1)
    parser.add_argument('-vocab_trunc_mode', default='size', 
                        help="How to truncate vocabulary size")
    
    parser.add_argument('-feat_vocab_size', type=int, default=1000)
    parser.add_argument('-feat_words_min_frequency', type=int, default=1)

    parser.add_argument('-share_vocab', action='store_true', default=False, 
                        help="Share source and target vocabulary")
    
    parser.add_argument('-pretrained', type=str, default='', help="choices: bert-base-uncased, gpt2, etc.")
    parser.add_argument('-pre_trained_vocab', default='',
                        help="Path to the pre-trained vocab file")
    parser.add_argument('-word_vec_size', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=32)
    
    ##### ========== Final Results Directory ========== #####
    parser.add_argument('-save_sequence_data', help="Output file for the prepared data")
    parser.add_argument('-save_graph_data', help="Output file for the prepared data")
