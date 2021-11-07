from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn import cluster

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.datasets import aggregate, statistics
from mad.plots import versus
from mad.plots import parity

from mad.datasets import load_data
from mad.ml import splitters, predict, feature_selectors

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run'
    points = None
    sampling = None

    # Load data
    data = load_data.test()
    df = data['frame']
    X = data['data']
    y = data['target']

    # ML setup
    scale = StandardScaler()
    inner_split = splitters.RepeatedClusterSplit(cluster.OPTICS, 5, 2)
    outer_split = splitters.RepeatedClusterSplit(cluster.OPTICS, 5, 2)
    selector = feature_selectors.no_selection()

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
    pipes = [lasso]

    # Evaluate
    predict.run(X, y, outer_split, pipes, save, seed)  # Perform ML
    aggregate.folds(save)  # Combine split data from directory recursively
    statistics.folds(save)  # Gather statistics from data
    parity.make_plots(save)  # Make parity plots
    versus.make_plots(save, points=points, sampling=sampling)


if __name__ == '__main__':
    main()
