from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso

from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration, err_in_err, nllh_versus
from mad.functions import poly

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run_lasso_diffusion'
    points = 100
    uq_func = poly
    uq_coeffs_start = [0.0, 1.1, 0.0]

    # Load data
    data = load_data.diffusion()
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

    # Do LASSO
    model = BaggingRegressor(base_estimator=Lasso())
    grid = {}
    grid['model__base_estimator__alpha'] = np.logspace(-5, 5, 11)
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    lasso = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain.builder(
                            lasso,
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
    parity.make_plots(save, 'pdf')  # Make parity plots
    calibration.make_plots(save, points, 'std', 'pdf')
    calibration.make_plots(save, points, 'stdcal', 'pdf')
    err_in_err.make_plots(save, points, 'mahalanobis')
    err_in_err.make_plots(save, points, 'pdf')
    nllh_versus.make_plots(save, points, 'mahalanobis')


if __name__ == '__main__':
    main()
