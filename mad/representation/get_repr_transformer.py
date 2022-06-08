import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from torch.optim import Adam
from mlm_pytorch import MLM

import torch.nn.functional as F

from mad.representation.trainer import fit_transformer

from mad.representation.datasets import SequenceSampling

from mad.representation.OOD_transformer import TransformerWrapper, Encoder,MyTokenEmbedding

cuda = torch.cuda.is_available()

def train_representation_transformer(data, label, dataset_name, loss = 'cosine'):
    epoch = 20

    if dataset_name == 'fried':
        seq_len = 100 # 100
        batch_size = 8
        epoch  = 20
        lr = 3e-3

        depth = 6 # too deep
        head = 8 # too much

    if dataset_name == 'diffu':
        seq_len = 100
        batch_size = 5
        epoch  = 20
        lr = 1e-3

        depth = 6 # too deep
        head = 8 # too much

    if dataset_name == 'supercond':
        seq_len = 100
        batch_size = 16
        epoch  = 20
        lr = 3e-4

        depth = 6 # OK?
        head = 8 # Ok?

    # dataloader setup
    dataset_len = data.shape[0]
    datasampler =  SequenceSampling( dataset_len = dataset_len , seq_len = seq_len)
    train_loader = torch.utils.data.DataLoader(datasampler, batch_size=batch_size, shuffle=True)
    # embedding table
    train_weight = data
    train_token_emb = nn.Embedding.from_pretrained(train_weight, freeze = True) # no grad, these embedding == original
    # model setup
    OODtransformer = TransformerWrapper(
        num_tokens = dataset_len, # vocab size
        max_seq_len = seq_len if dataset_len > seq_len else dataset_len,
        attn_layers = Encoder(
            dim = data.shape[1], # 512,
            depth = depth, # default 6
            heads = head
        ),
        my_embedder = train_token_emb
    )

    # plugin the language model into the MLM trainer
    trainer = MLM(
        OODtransformer ,
        mask_token_id = 2,          # the token id reserved for masking
        pad_token_id = 0,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.0,        # 0% probability that token will be masked, but included in loss, as detailed in the epaper
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    )

    optimizer = Adam(trainer.parameters(), lr=lr)
    interval = 10 # useless

    fit_transformer(train_loader, trainer, optimizer, epoch, cuda, interval)

    return OODtransformer
