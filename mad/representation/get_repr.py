import pandas as pd
import numpy as np

import torch

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from mad.representation.networks import EmbeddingNetLarge, EmbeddingNetMiddle, EmbeddingNetSmall, SiameseNet
from mad.representation.trainer import fit
from mad.representation.losses import EasyPositiveSemiHardNegativeLossCosine, EasyPositiveHardNegativeLossCosine, SCTLossCosine,EasyPositiveHardNegativeLossMaha
from mad.representation.datasets import BatchPairSampling

cuda = torch.cuda.is_available()

def train_representation(data, label, dataset_name):


    # loss_fn = EasyPositiveSemiHardNegativeLossCosine( )
    loss_fn = EasyPositiveHardNegativeLossCosine()
    # loss_fn =  SCTLossCosine( method = 'sct', lam = 1)

    # loss_fn = EasyPositiveHardNegativeLossMaha()

    if dataset_name == 'fried':

        lr = 0.5*1e-3

        n_epochs = 50
        hidden_size = 10
        embedding_size = 5
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 16#8
        net = EmbeddingNetMiddle(input_size, hidden_size, embedding_size)

    if dataset_name == 'diffu':
        lr = 1*1e-4 # 1e-3

        n_epochs = 25 # 30
        hidden_size = 100
        embedding_size = 20
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 8 # 8
        net = EmbeddingNetSmall(input_size, hidden_size, embedding_size)

    if dataset_name == 'supercond':
        lr = 1*1e-3# 1*1e-4 # 0.5*1e-3

        n_epochs = 60 # 50 # 40
        hidden_size = 100
        embedding_size = 20
        input_size = data.shape[1]
        interval = 10 # useless
        batch_size = 45 # 35 25
        net = EmbeddingNetLarge(input_size, hidden_size, embedding_size)

    siamese_net = SiameseNet(  embedding_net = net )
    model = siamese_net
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    datasampler = BatchPairSampling( dataset = data , labels= label)
    loader = torch.utils.data.DataLoader(datasampler, batch_size=batch_size, shuffle=True)

    fit(loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, interval)

    return model
