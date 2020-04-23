import argparse


def add_data_options(parser):
     ## Data options
     parser.add_argument('-sequence_data', required=True)
     parser.add_argument('-graph_data', required=True)
     parser.add_argument('-train_dataset', required=True)
     parser.add_argument('-valid_dataset', required=True)

     # Test options
     parser.add_argument('-max_token_src_len', type=int, default=300)
     parser.add_argument('-max_token_tgt_len', type=int, default=50)
     parser.add_argument('-accum_count', type=int, nargs='+', default=[1],
                         help="Accumulate gradient this many times. "
                              "Approximately equivalent to updating batch_size * accum_count batches at once. "
                              "Recommended for Transformer.")


def add_model_options(parser):
     ## checkpoint
     parser.add_argument('-checkpoint', type=str, default='',
                         help="Path of trained model to do further training")

     ## Model options
     parser.add_argument('-epoch', type=int, default=10)
     parser.add_argument('-batch_size', type=int, default=64)

     parser.add_argument('-answer', default=False, action='store_true')
     parser.add_argument('-node_feature', default=False, action='store_true',
                         help="whether to incorporate node feature information")
     parser.add_argument('-feature', default=False, action='store_true',
                         help="whether to incorporate word feature information")
     parser.add_argument('-dec_feature', type=int, default=0,
                         help="""Number of features directly sent to the decoder""")
     
     parser.add_argument('-pretrained', type=str, default='', help="choices: bert-base-uncased, gpt2, etc.")
     parser.add_argument('-pre_trained_vocab', action='store_true', default=False,
                         help="whether to use a pretrained word vector for embedding")
          
     parser.add_argument('-d_word_vec', type=int, default=512,
                         help="hidden size of word-vector in embedding layer")
     parser.add_argument('-d_feat_vec', type=int, default=32,
                         help="size of feature embedding vector")
     
     parser.add_argument('-sparse', type=int, default=1)

     parser.add_argument('-d_seq_enc_model', type=int, default=512,
                         help="hidden_size * num_directions of vector in RNN encoder")
     parser.add_argument('-d_graph_enc_model', type=int, default=512,
                         help="hidden_size of vector in graph encoder")
     parser.add_argument('-d_dec_model', type=int, default=512,
                         help="hidden size of vector in decoder")
     parser.add_argument('-n_seq_enc_layer', type=int, default=1, help="number of rnn encoder layers")
     parser.add_argument('-n_graph_enc_layer', type=int, default=3, help="number of graph encoder layers")
     parser.add_argument('-n_dec_layer', type=int, default=1, help="number of decoder layer")

     parser.add_argument('-n_head', type=int, default=8, 
                         help="number of heads in multi-head self-attention mechanism")
     parser.add_argument('-d_inner', type=int, default=2048,
                         help="size of inner vector of the 1st layer in feed forward network")
     parser.add_argument('-d_k', type=int, default=64,
                         help="size of attention vector")
     parser.add_argument('-d_v', type=int, default=64,
                         help="size of vectors which will be used to calculate weighted context vector")
     
     parser.add_argument('-alpha', type=float, default=0.1, help="for LeakyReLu")

     parser.add_argument('-brnn', default=False, action='store_true', 
                         help="whether to use a bidirectional RNN in encoder")
     parser.add_argument('-input_feed', type=bool, default=1, 
                         help="whether to incorporate encoder-hidden-state directly into RNN in decoder")
     parser.add_argument('-enc_rnn', type=str, choices=['gru', 'lstm'], default='gru')
     parser.add_argument('-dec_rnn', type=str, choices=['gru', 'lstm'], default='gru')
     
     parser.add_argument('-defined_slf_attn_mask', default='', type=str,
                         help="Path to the self-attention matrix for each sequences")

     parser.add_argument('-copy', action='store_true', default=False,
                         help="""Use a copy mechanism""")
     parser.add_argument('-coverage', action='store_true', default=False,
                         help="""Use a coverage mechanism""")
     parser.add_argument('-coverage_weight', type=float, default=1.0, 
                         help="""Weight of the loss of coverage mechanism in final total loss""")
     
     parser.add_argument('-slf_attn', action='store_true', default=False,
                         help="source self-attention encoding")

     parser.add_argument('-maxout_pool_size', type=int, default=2,
                         help='Pooling size for MaxOut layer.')
               
     parser.add_argument('-layer_attn', default=False, action='store_true',
                         help="""Wether to add a universal cross attention 
                              on all outputs of the encoder layers to generate 
                              a aggregate context of all layers""")
     
     parser.add_argument('-proj_share_weight', action='store_true',
                         help="whether to share weight between embedding and final projecting layers")


def add_train_options(parser):
     # log
     parser.add_argument('-log_home', required=True, help="""log home""")

     parser.add_argument('-save_model', default=None)
     parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

     parser.add_argument('-valid_steps', type=int, default=500,
                         help="Number of interval steps between two adjacent times of evaluation")

     parser.add_argument('-logfile_train', default='',
                         help="Path to save loss and evaluation reports on training data")
     parser.add_argument('-logfile_dev', default='',
                         help="Path to save loss and evaluation reports on validation data")
     
     # BLeU-4
     parser.add_argument('-translate_ppl', type=float, default=40,
                         help="Start to calculate BLeU4 on validation data when its PPL reach this number.")
     parser.add_argument('-translate_steps', type=int, default=2500, 
                         help="Number of interval steps between two adjacent times of translation")

     # training trick
     parser.add_argument('-training_mode', required=True, choices=['unify', 'generate', 'classify'])
     parser.add_argument('-n_warmup_steps', type=int, default=4000)
     parser.add_argument('-dropout', type=float, default=0.1)
     parser.add_argument('-attn_dropout', type=float, default=0.1)

     parser.add_argument('-curriculum', type=int, default=1,
                         help="""For this many epochs, order the minibatches based
                         on source sequence length. Sometimes setting this to 1 will
                         increase convergence speed.""")
     parser.add_argument('-extra_shuffle', action="store_true",
                         help="""By default only shuffle mini-batch order; when true,
                         shuffle and re-assign mini-batches""")

     # learning rate
     parser.add_argument('-optim', default='sgd', 
                         choices=['sgd', 'adagrad', 'adadelta', 'adam', 'sparseadam', 'fusedadam'],
                         help="Optimization method.")
     parser.add_argument('-learning_rate', type=float, default=1.0,
                         help="Starting learning rate. "
                              "Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001")
     parser.add_argument('-decay_method', type=str, default='', choices=['noam', 'noamwd', 'rsqrt', ''],
                         help="Use a custom decay rate.")
     parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                         help="If update_learning_rate, decay learning rate by "
                              "this much if steps have gone past "
                              "start_decay_steps")
     parser.add_argument('-decay_steps', type=int, default=500,
                         help="Decay every decay_steps")
     parser.add_argument('-start_decay_steps', type=int, default=1000,
                         help="Start decaying every decay_steps after start_decay_steps")
     parser.add_argument('-max_grad_norm', type=float, default=5,
                         help="If the norm of the gradient vector exceeds this, "
                              "renormalize it to have the norm equal to max_grad_norm")
     parser.add_argument('-max_weight_value', type=float, default=15,
                         help="If the norm of the gradient vector exceeds this, "
                              "renormalize it to have the norm equal to max_grad_norm")
     parser.add_argument('-decay_bad_cnt', type=int, default=3)

     # GPU
     parser.add_argument('-gpus', default=[], nargs='+', type=int,
                         help="Use CUDA on the listed devices.")

     parser.add_argument('-log_interval', type=int, default=100,
                         help="logger.info stats at this interval.")

     parser.add_argument('-seed', type=int, default=-1,
                         help="""Random seed used for the experiments
                         reproducibility.""")
     parser.add_argument('-cuda_seed', type=int, default=-1,
                         help="""Random CUDA seed used for the experiments
                         reproducibility.""")
     
     # translate
     parser.add_argument('-eval_batch_size', type=int, default=16)
     parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
     parser.add_argument('-n_best', type=int, default=1,
                         help="""If verbose is set, will output the n_best
                         decoded sentences""")