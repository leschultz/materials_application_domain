from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import *
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np

import random
import os

from mad.functions import parallel, llh, set_llh


def distance_link(X_train, X_test, dist_type, append_name=''):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        append_name = The string to append to name of distance metric.

    ouputs:
        dists = A dictionary of distances.
    '''

    dists = {}
    if dist_type == 'mahalanobis':
        # Get the inverse of the covariance matrix from training
        if X_train.shape[1] < 2:

            vals = np.empty(X_test.shape[0])
            dists[append_name+dist_type+'_mean'] = vals
            dists[append_name+dist_type+'_max'] = vals
            dists[append_name+dist_type+'_min'] = vals

        else:
            vi = np.linalg.inv(np.cov(X_train.T))
            dist = cdist(X_train, X_test, dist_type, VI=vi)

            dists[append_name+dist_type+'_mean'] = np.mean(dist, axis=0)
            dists[append_name+dist_type+'_max'] = np.max(dist, axis=0)
            dists[append_name+dist_type+'_min'] = np.min(dist, axis=0)

    elif dist_type == 'pdf':

        bw = estimate_bandwidth(X_train)
        model = KernelDensity(kernel='gaussian', bandwidth=bw)
        model.fit(X_train)

        log_dist = model.score_samples(X_test)

        dists[append_name+dist_type] = np.exp(log_dist)
        dists[append_name+'log'+dist_type] = log_dist

    else:
        dist = cdist(X_train, X_test, dist_type)

        dists[append_name+dist_type+'_mean'] = np.mean(dist, axis=0)
        dists[append_name+dist_type+'_max'] = np.max(dist, axis=0)
        dists[append_name+dist_type+'_min'] = np.min(dist, axis=0)

    return dists


def distance(X_train, X_test):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    distance_list = [
                'pdf',
                'mahalanobis',
                'euclidean',
                'minkowski',
                'cityblock',
                'seuclidean',
                'sqeuclidean',
                'cosine',
                'correlation',
                'chebyshev',
                'canberra',
                'braycurtis',
                'sokalsneath',
                ]

    matrix_decomp_methods = [
                             PCA(),
                             SparsePCA(),
                             KernelPCA(n_components=X_train.shape[1]),
                             SparsePCA(),
                             IncrementalPCA(),
                             MiniBatchSparsePCA(),
                             ]

    dists = {}
    for distance in distance_list:

        # Compute regular distances
        dists.update(distance_link(X_train, X_test, distance))

        # Compute transformed distances
        for cur_method in matrix_decomp_methods:
            cur_method.fit(X_train)  # Refit object
            X_test_transformed = cur_method.transform(X_test)
            name = cur_method.__class__.__name__+'_'
            dists.update(distance_link(
                                       X_train,
                                       X_test_transformed,
                                       distance,
                                       name
                                       ))

    return dists


def inner(indx, X, y, pipes, save, groups=None):
    '''
    The inner loop from nested cross validation.
    '''

    # The splitting indexes and assigned id
    indx, count = indx

    tr_indx, te_indx = indx
    X_train, X_test = X[tr_indx], X[te_indx]
    y_train, y_test = y[tr_indx], y[te_indx]

    if groups is not None:
        g_train, g_test = groups[tr_indx], groups[te_indx]

    # Do ML for each outer fold
    trains = []
    tests = []
    for pipe in pipes:

        train = {}
        test = {}

        if groups is None:
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train, g_train)

        pipe_best = pipe.best_estimator_
        pipe_best_scaler = pipe_best.named_steps['scaler']
        pipe_best_select = pipe_best.named_steps['select']
        pipe_best_model = pipe_best.named_steps['model']

        model_type = pipe_best_model.__class__.__name__
        scaler_type = pipe_best_scaler.__class__.__name__
        split_type = pipe.cv.__class__.__name__

        X_test_trans = pipe_best_scaler.transform(X_test)
        X_train_trans = pipe_best_scaler.transform(X_train)
        X_train_select = pipe_best_select.transform(X_train_trans)
        X_test_select = pipe_best_select.transform(X_test_trans)

        n_features = X_train_select.shape[-1]

        # If model is ensemble regressor
        ensemble_methods = ['RandomForestRegressor', 'BaggingRegressor']
        if model_type in ensemble_methods:
            y_test_pred = pipe_best.predict(X_test)
            y_train_pred = pipe_best.predict(X_train)

            # Ensemble predictions with correct feature set
            pipe_estimators = pipe_best_model.estimators_
            std_test = [i.predict(X_test_select) for i in pipe_estimators]
            std_train = [i.predict(X_train_select) for i in pipe_estimators]

            std_test = np.std(std_test, axis=0)
            std_train = np.std(std_train, axis=0)

            train['std'] = std_train
            test['std'] = std_test

        # If model is gaussian process regressor
        elif model_type == 'GaussianProcessRegressor':
            y_test_pred, std_test = pipe_best.predict(X_test, return_std=True)
            y_train_pred, std_train = pipe_best.predict(
                                                        X_train,
                                                        return_std=True
                                                        )

            train['std'] = std_train
            test['std'] = std_test

        else:
            y_test_pred = pipe_best.predict(X_test)
            y_train_pred = pipe_best.predict(X_train)

        # Add std calibrated here and include as distance metric
        if 'std' in test.keys():

            # Calibration
            a, b = set_llh(std_train, y_train, y_train_pred, [0, 1])
            std_train_cal = a*std_train+b
            std_test_cal = a*std_test+b

            # Log likelihoods
            llh_vals_train = llh(std_train, y_train-y_train_pred, [a, b])
            llh_vals_test = llh(std_test, y_test-y_test_pred, [a, b])

            train['std_cal'] = std_train_cal
            train['loglikelihood'] = llh_vals_train
            test['std_cal'] = std_test_cal
            test['loglikelihood'] = llh_vals_test

        # Calculate distances from test set cases to traning set
        dists_test = {}
        for key, value in distance(X_train_select, X_test_select).items():
            if key in dists_test:
                dists_test[key] += list(value)
            else:
                dists_test[key] = list(value)

        # Calculate the distances from train set cases to training set
        dists_train = {}
        for key, value in distance(X_train_select, X_train_select).items():
            if key in dists_train:
                dists_train[key] += list(value)
            else:
                dists_train[key] = list(value)

        # Assign values that should be the same
        train['pipe'] = test['pipe'] = pipe
        train['model'] = test['model'] = model_type
        train['scaler'] = test['scaler'] = scaler_type
        train['features'] = test['features'] = n_features
        train['splitter'] = test['splitter'] = split_type
        train['split_id'] = test['split_id'] = count

        # Training data
        train['y'] = y_train
        train['y_pred'] = y_train_pred
        train['index'] = tr_indx

        # Testing data
        test['y'] = y_test
        test['y_pred'] = y_test_pred
        test['index'] = te_indx

        train.update(dists_train)
        test.update(dists_test)

        train = pd.DataFrame(train)
        test = pd.DataFrame(test)

        trains.append(train)
        tests.append(test)

    train_name = os.path.join(
                              save,
                              'train_split_{}.csv'.format(count)
                              )

    test_name = os.path.join(
                             save,
                             'test_split_{}.csv'.format(count)
                             )
    pd.concat(trains).to_csv(train_name, index=False)
    pd.concat(tests).to_csv(test_name, index=False)


def outer(split, pipes, X, y, save, groups=None):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        split = The splitting method.
        pipes = The machine learning pipeline.
        X = The feature matrix.
        y = The target values.
        save = The directory to save values.
        groups = The class to group by.
    '''

    # Output directory creation
    save = os.path.join(save, 'splits')
    os.makedirs(save, exist_ok=True)

    # Gather split data in parallel
    if groups is not None:
        splits = list(split.split(X, y, groups))
    else:
        splits = list(split.split(X))

    counts = list(range(len(splits)))
    parallel(
             inner,
             list(zip(splits, counts)),
             X=X,
             y=y,
             pipes=pipes,
             save=save,
             groups=groups,
             )


def run(X, y, split, pipes, save, seed=1, groups=None):
    '''
    Define the machine learning workflow with nested cross validation
    for gaussian process regression and random forest.
    '''

    # Setting seed for reproducibility
    np.random.seed(seed)
    np.random.RandomState(seed)
    random.seed(seed)

    # Nested CV
    outer(split, pipes, X, y, save, groups)
