import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import torch.optim as custom_optim

# from simple_nmt.data_loader import DataLoader
# import nmt.data_loader as data_loader
from data_loader import DataLoader
import data_loader as data_loader


# from simple_nmt.models.seq2seq import Seq2Seq
# from simple_nmt.models.transformer import Transformer
# from simple_nmt.models.rnnlm import LanguageModel
from models.seq2seq import Seq2Seq
# from models.transformer import Transformer
# from models.rnnlm import LanguageModel

# from simple_nmt.trainer import SingleTrainer
# from simple_nmt.rl_trainer import MinimumRiskTrainingEngine
# from simple_nmt.trainer import MaximumLikelihoodEstimationEngine

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'

    )

    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex:train.en --> train)'
    )

    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extension. (ex: valid.en --> valid)'
    )

    p.add_argument(
        '--lang',
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )

    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )

    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )

    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )

    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )

    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )

    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )

    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )

    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )

    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=.5,
        help='Threshold for gradient clipping. Default=%(default)s'
    )

    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s'
    )

    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s'
    )

    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s'
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments shuold be changed.'
    )

    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.'
    )

    p.add_argument(
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )

    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )

    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )

    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        hel='Maximum number of tokens to calculate BLEU for reinforce learning. Default=%(default)s'
    )

    p.add_argument(
        '--rl_reward',
        type=str,
        default='gleu',
        help='Metohd name to use as reward function for RL training. Default=%(default)s'
    )

    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer'
    )

    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s'
    )

    config = p.parse_args()

    return config

def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,
            config.hidden_size,
            output_size,
            n_splits=config.n_splits,
            n_enc_blocks=config.n_layers,
            n_dec_blocks=config.n_layers,
            dropout_p=config.
        )


