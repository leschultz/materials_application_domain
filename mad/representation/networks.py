import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingNetMiddle(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size):
        super(EmbeddingNetMiddle, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # size of hidden layer
        self.embedding_size = embedding_size # size of last layer as embedding

        # have hidden layer, last layer as embedding
        self.fc = nn.Sequential(nn.Linear(self.input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, self.embedding_size)
                                )

    def forward(self, x):
        x = x.float()
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNetSmall(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size):
        super(EmbeddingNetSmall, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # size of hidden layer
        self.embedding_size = embedding_size # size of last layer as embedding

        # have hidden layer, last layer as embedding
        self.fc = nn.Sequential(nn.Linear(self.input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, self.embedding_size)
                                )

    def forward(self, x):
        x = x.float()
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNetLarge(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size):
        super(EmbeddingNetLarge, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # size of hidden layer
        self.embedding_size = embedding_size # size of last layer as embedding

        # have hidden layer, last layer as embedding
        self.fc1 = nn.Sequential(nn.Linear(self.input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                )
        # self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.embedding_size)


    def forward(self, x):
        x = x.float()
        output1 = self.fc1(x)
        return self.fc2(output1)

    def get_embedding(self, x):
        emb = self.forward(x)
        # emb = self.fc(x)
        return emb

class TripletNet(nn.Module):

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        # using the same network to output embedding for 3 inputs
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiameseNet(nn.Module):

    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        # using the same network to output embedding for 3 inputs
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_model, dim=20, prev_dim=10):
        """
        base_model:  encoder
        dim:         representation dimension (default: 20 for super/diffu, friedman 5)
        pred_dim:    dim of the bottle neck
        """
        super(SimSiam, self).__init__()

        self.encoder = base_model

        # projection head
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                        nn.BatchNorm1d(prev_dim),
                        nn.ReLU(inplace=True), # first layer
                        nn.Linear(prev_dim, prev_dim, bias=False),
                        nn.BatchNorm1d(prev_dim),
                        nn.ReLU(inplace=True), # second layer
                        nn.Linear(prev_dim, prev_dim, bias=False),
                        nn.ReLU(inplace=True), # third layer
                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # predictor
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(prev_dim, dim)) # output layer

    def forward(self, x1,x2):
        """
        Input:
            x1: first views of inputs
            x2: second views of inputs
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def get_embedding(self, x):
        return self.encoder(x)
