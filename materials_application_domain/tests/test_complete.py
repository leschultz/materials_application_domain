from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn import cluster

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.datasets import load_data, aggregate, statistics
from mad.plots import kde, parity, versus, calibration
from mad.ml import splitters, predict, feature_selectors

import numpy as np
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test ml workflow
        '''

        seed = 14987
        save = './test'
        points = 15
        sampling = 'equal'

        # Load data
        data = load_data.test()
        df = data['frame']
        X = data['data']
        y = data['target']

        # ML setup
        scale = StandardScaler()
        inner_split = splitters.RepeatedClusterSplit(cluster.OPTICS, 5, 2)
        outer_split = splitters.NoSplit()
        selector = feature_selectors.no_selection()

        # Gaussian process regression
        kernel = RBF()
        model = GaussianProcessRegressor()
        grid = {}
        grid['model__alpha'] = np.logspace(-2, 2, 5)
        grid['model__kernel'] = [RBF()]
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        gpr = GridSearchCV(pipe, grid, cv=inner_split)

        # Random forest regression
        model = RandomForestRegressor()
        grid = {}
        grid['model__n_estimators'] = [100]
        grid['model__max_features'] = [None]
        grid['model__max_depth'] = [None]
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        rf = GridSearchCV(pipe, grid, cv=inner_split)

        # Do LASSO
        model = BaggingRegressor(base_estimator=Lasso())
        grid = {}
        grid['model__base_estimator__alpha'] = np.logspace(-5, 5, 11)
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        lasso = GridSearchCV(pipe, grid, cv=inner_split)

        # Make pipeline
        pipes = [gpr, rf, lasso]

        # Evaluate
        predict.run(X, y, outer_split, pipes, save, seed)  # Perform ML
        aggregate.folds(save)  # Combine split data from directory recursively
        statistics.folds(save)  # Gather statistics from data
        parity.make_plots(save)  # Make parity plots
        versus.make_plots(save, points, sampling)
        calibration.make_plots(save, points, sampling)
        kde.make_plots(df, save)  # Global KDE plots

        # Clean directory
        shutil.rmtree(save)


if __name__ == '__main__':
    unittest.main()
