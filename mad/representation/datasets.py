import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch

class TripletSampling(Dataset):
    """
    Dataset should be raw feature (not target var included)
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: UNIMPLEMENTED
    """

    def __init__(self, dataset, labels):

        self.dataset = dataset
        self.labels = np.array(labels) # 1 indicates positive class, 0 for negative class

        self.labels_set = set(self.labels)

        # key: label, value: a list of indicies
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}


    def __getitem__(self, index):
        """given the index of anchor, return (anchor, pos, neg) """
        anchor, anchor_label = self.dataset[index], self.labels[index]
        positive_index = index
        while positive_index == index: # make sure the pos is not the anchor
            positive_index = np.random.choice(self.label_to_indices[anchor_label])

        negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        positive = self.dataset[positive_index]
        negative = self.dataset[negative_index]

        return (anchor, positive, negative)

    def __len__(self):
        return len(self.dataset)

class BatchPairSampling(Dataset):
    """
    Dataset should be raw feature (not target var included)
    Train: return a batch of pairs. Each pair: (positive, negative)
    Test: UNIMPLEMENTED
    """

    def __init__(self, dataset, labels):


        self.dataset = dataset
        self.labels = np.array(labels) # 1 indicates positive class, 0 for negative class

        self.labels_set = set(self.labels)

        # key: label, value: a list of indicies
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}


    def __getitem__(self, index):
        """given the index of sample, determine it's cls, return (pos, neg) """
        # we don't know whether sample_label is pos or neg
        sample, sample_label = self.dataset[index], self.labels[index]
        # find a sample of an oppositive class(label)
        opposite_label = np.random.choice(list(self.labels_set - set([sample_label])))
        opposite_index = np.random.choice(self.label_to_indices[opposite_label])
        opposite_sample = self.dataset[opposite_index]

        positive = None
        negative = None
        if sample_label == 1:
            positive = sample
            negative = opposite_sample
        else:
            positive = opposite_sample
            negative = sample
        return (positive, negative)


    def __len__(self):
        return len(self.dataset)

class SimSiamSampling(Dataset):

    def __init__(self, dataset, labels, mean, variance, select):


        self.dataset = dataset
        self.dim = dataset.shape[1]
        self.labels = np.array(labels) # 1 indicates positive class, 0 for negative class
        self.labels_set = set(self.labels)
        self.mean = mean
        self.variance = variance
        self.select = select # how many contiguous feature I want to select for augmentation?

    def __getitem__(self, index):
        """given the index of sample, determine it's cls, return (pos, neg) """
        sample, sample_label = self.dataset[index], self.labels[index]
        x1,x2 = self.augmentation(sample)
        return (x1,x2)


    def __len__(self):
        return len(self.dataset)

    def augmentation(self, x):
        # at this phase, still keeps the same dimension
        x1, x2 = self.add_noise(x, self.mean, self.variance)
        # random "croping" == a contiguous feature selection
        if (self.dim - self.select) >= 2:
            x1, x2 = self.cropping(x)
        return x1,x2

    def add_noise(self, x, mean=[0,0], variance = [5,5]):

        mean1,mean2 = random.choices(mean, k = 2)
        variance1, variance2 = random.choices(variance, k = 2)
        noise1 = np.random.normal(mean1, variance1, self.dim )
        noise2 = np.random.normal(mean2, variance2, self.dim )

        return x + noise1, x + noise2
    def cropping(self, x ):
        delta = self.dim - self.select
        start1, start2 = random.choices( range(delta - 1), k =2)
        return x[start1:start1+self.select], x[start2:start2+self.select]


class SequenceSampling(Dataset):

    def __init__(self, dataset_len, seq_len):

        self.dataset_len = dataset_len
        self.seq_len = seq_len # length of sequence we construct, contains token (idex)
        self.seq_range = dataset_len # range of token (idx) that will be put into a sequence

    def __getitem__(self, index):
#         print ( f"index {index}" )
        """given the index of sample, make sure it is inside the sequence, then sample from the rest """
#         assert index < self.seq_range, f"Index {index} Out of Bounds in dataset __getitem__\n"
        # sample without replacement
        seq = [index] + random.sample( list( set(range(self.seq_range)) - {index} ), self.seq_len-1)
        result =  torch.tensor(seq)
        if sum( np.array(seq) >= self.dataset_len ) > 0:
            assert False
        return result


    def __len__(self):
        return self.dataset_len
