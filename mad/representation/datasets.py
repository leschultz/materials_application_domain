import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

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