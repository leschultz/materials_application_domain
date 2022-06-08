import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from mad.representation.networks import EmbeddingNetLarge, EmbeddingNetMiddle, EmbeddingNetSmall, SimSiam
from mad.representation.trainer import fit, fit2

from mad.representation.datasets import SimSiamSampling

cuda = torch.cuda.is_available()

def train_representation_simsiam(data, label, dataset_name, loss = 'cosine'):

    criterion = nn.CosineSimilarity(dim=1)

    if dataset_name == 'fried':
        select_dim = 3
        batch_size = 8
        lr = 0.01 * batch_size / 256
        n_epochs = 50
        hidden_size = 10
        embedding_size = 5
        input_size = select_dim
        interval = 10 # useless

        mean = [0.2,-0.2]
        variance = [5,5]

        net = EmbeddingNetMiddle(input_size, hidden_size, embedding_size)

    if dataset_name == 'diffu':
        select_dim = 22
        batch_size = 10
        lr = 0.05 * batch_size / 256

        n_epochs = 50 # 35 # 25
        hidden_size = 100
        embedding_size = 20
        input_size = select_dim
        interval = 15 # useless
        net = EmbeddingNetSmall(input_size, hidden_size, embedding_size)
        # net = EmbeddingNetMiddle(input_size, hidden_size, embedding_size)

        mean = [-0.2, 0.1]
        variance = [5,5] # [5,5]

    if dataset_name == 'supercond':
        select_dim = 22
        batch_size = 35
        lr = 0.05 * batch_size / 256
        n_epochs = 50
        hidden_size = 100
        embedding_size = 20
        input_size = select_dim
        interval = 10 # useless
        net = EmbeddingNetLarge(input_size, hidden_size, embedding_size)

        mean = [-0.5, 0.2]
        variance = [3,5]

    model = SimSiam(base_model = net,
                dim = embedding_size, # embedding layer dimension, input dimension of predictor
                prev_dim= embedding_size//2 if embedding_size//2 >=5 else 3 # bottleneck of predictor
               )
    model = model.float()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    datasampler = SimSiamSampling(
                                dataset = data,
                                labels = label,
                                mean = mean,
                                variance = variance,
                                select = select_dim)
    loader = torch.utils.data.DataLoader(datasampler, batch_size=batch_size, shuffle=True, drop_last=True)
    # fit2 is specific for simple siamese network
    fit2(loader, model, criterion, optimizer, scheduler, n_epochs, cuda, interval)

    return model
