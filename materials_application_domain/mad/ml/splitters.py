from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.utils import resample

import pandas as pd
import numpy as np
import random


class NoSplit:
    '''
    Class to not split data
    '''

    def get_n_splits(self):
        '''
        Should only have one split.
        '''

        return 1

    def split(self, X, y=None, groups=None):
        '''
        The training and testing indexes are the same.
        '''

        indx = range(X.shape[0])

        yield indx, indx


class BootstrappedLeaveOneGroupOut:
    '''
    Custom splitting class which with every iteration of n_repeats it will
    bootstrap the dataset with replacement and leave every group out once
    with a given class column. (Special thanks to Angelo Cortez and Yiqi Wang
    for development).
    '''

    def __init__(self, n_repeats, groups, *args, **kwargs):
        '''
        inputs:
            n_repeats = The number of times to apply splitting.
            groups =  np.array of group classes for the dataset.
        '''

        self.groups = groups
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the O(N) number of splits.
        '''

        self.groups = groups
        self.n_splits = self.n_repeats*len(set(list(self.groups)))

        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        '''
        For every iteration, bootstrap the original dataset, and leave
        every group out as the testing set one time.
        '''

        grouping_df = pd.DataFrame(self.groups, columns=['group'])
        unique_groups = list(set(self.groups))

        # Resample self.n_repeats times
        for rep in range(self.n_repeats):

            sample = resample(grouping_df)  # Boostrap
            for unique_group in unique_groups:

                # Check first if the group is in the bootstrapped dataset
                if len(sample[sample['group'] == unique_group]) > 0:

                    test = sample[sample['group'] == unique_group]
                    train = sample[sample['group'] != unique_group]

                    test = test.index.tolist()
                    train = train.index.tolist()

                    yield train, test


class RepeatedClusterSplit:
    '''
    Custom splitting class which pre-clusters data and then splits
    to folds.
    '''

    def __init__(self, clust, n_splits, n_repeats, *args, **kwargs):
        '''
        inputs:
            clust = The class of cluster from Scikit-learn.
            n_splits = The number of splits to apply.
            n_repeats = The number of times to apply splitting.
        '''

        self.clust = clust(*args, **kwargs)

        # Make sure it runs in serial
        if hasattr(self.clust, 'n_jobs'):
            self.clust.n_jobs = 1

        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_splits*self.n_repeats

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        self.clust.fit(X)  # Do clustering

        # Get splits based on cluster labels
        df = pd.DataFrame(X)
        df['cluster'] = self.clust.labels_
        cluster_order = list(set(self.clust.labels_))
        random.shuffle(cluster_order)
        df = df.sample(frac=1)

        # Randomize cluster order
        df = [df.loc[df['cluster'] == i] for i in cluster_order]
        df = pd.concat(df)

        s = np.array_split(df, self.n_splits)  # Split
        range_splits = range(self.n_splits)

        # Do for requested repeats
        for rep in range(self.n_repeats):

            s = [i.sample(frac=1) for i in s]  # Shuffle
            for i in range_splits:

                te = s[i]  # Test
                tr = pd.concat(s[:i]+s[i+1:])  # Train

                # Get the indexes
                train = tr.index.tolist()
                test = te.index.tolist()

                yield train, test


class RepeatedPDFSplit:
    '''
    Custom splitting class which groups data on a multivariate probability
    distribution function and then splits to folds. Folds should have
    least probable cases.
    '''

    def __init__(self, frac, n_repeats, *args, **kwargs):
        '''
        inputs:
            frac = The fraction of the split.
            n_repeats = The number of times to apply splitting.
        '''

        self.frac = frac
        self.n_repeats = n_repeats

    def get_n_splits(self, X=None, y=None, groups=None):
        '''
        A method to return the number of splits.
        '''

        return self.n_repeats

    def split(self, X, y=None, groups=None):
        '''
        Cluster data, randomize cluster order, randomize case order,
        and then split into train and test sets self.reps number of times.

        inputs:
            X = The features.
        outputs:
            A generator for train and test splits.
        '''

        for i in range(self.n_repeats):

            # Sample with replacement
            bootstrapped_df = resample(X)

            bw = estimate_bandwidth(X)
            model = KernelDensity(kernel='gaussian', bandwidth=bw)
            model.fit(X)

            dist = model.score_samples(X)  # Natural log distance

            df = {'dist': dist, 'index': list(range(bootstrapped_df.shape[0]))}
            df = pd.DataFrame(df)
            df.sort_values(by='dist', inplace=True)

            split = int(df.shape[0]*self.frac)
            test = df[split:].index.tolist()
            train = df[:split].index.tolist()

            yield train, test
