import torch.nn as nn
import torch.nn.functional as F

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