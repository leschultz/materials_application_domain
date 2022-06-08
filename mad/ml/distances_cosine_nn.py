from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from scipy.spatial.distance import cdist
from sklearn.decomposition import *
from sklearn.svm import OneClassSVM

import numpy as np

def distance_link(
                  X_train,
                  X_test,
                  model,
                  dist_type,
                  append_name='',
                  y_train=None,
                  y_test=None
                  ):
    '''
    Get the distances based on a metric.
    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        append_name = The string to append to name of distance metric.
        y_train = The training target when applicable.
        y_test = The testing target when applicable.
    ouputs:
        dists = A dictionary of distances.
    '''

    dists = {}

    if dist_type == 'cosine_nn':

        if X_train.shape[1] < 2:
            vals = np.empty(X_test.shape[0])
            dists[append_name+dist_type] = vals
        else:
            score = model.predict_proba(X_test)[:,0] # we take OOD probability score
            # assert False, f"Score shape: {score.shape}\n"
            dists[append_name+dist_type] = score

    else:
        assert False

    return dists


def distance(X_train, X_test, model, y_train=None, y_test=None):
    '''
    Determine the distance from set X_test to set X_train.
    '''
    # For development
    distance_list = [
                     'cosine_nn',
                     ]

    dists = {}
    for distance in distance_list:

        # Compute regular distances
        dists.update(distance_link(
                                   X_train,
                                   X_test,
                                   model,
                                   distance,
                                   y_train=y_train,
                                   y_test=y_test
                                   ))

    return dists
