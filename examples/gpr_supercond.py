from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration
from mad.functions import poly

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run_gpr_diffusion'
    points = 15
    uq_func = poly
    uq_coeffs_start = [0.0, 1.1, 0.0]

    # Load data
    data = load_data.super_cond()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = data['class_name']

    # Splitters
    top_split = splitters.BootstrappedLeaveOneGroupOut(n_repeats=2, groups=d)
    mid_split = RepeatedKFold(n_splits=5, n_repeats=2)
    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Gaussian Process Regression
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
    gpr = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain.builder(
                            gpr,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save,
                            seed=seed,
                            uq_func=uq_func,
                            uq_coeffs_start=uq_coeffs_start
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data

    # Make parity plots
    parity.make_plots(save, 'mahalanobis')
    parity.make_plots(save, 'attention_metric')

    # Make calibration plots
    calibration.make_plots(save, points, 'std', 'mahalanobis')
    calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    calibration.make_plots(save, points, 'std', 'attention_metric')
    calibration.make_plots(save, points, 'stdcal', 'attention_metric')


if __name__ == '__main__':
    main()