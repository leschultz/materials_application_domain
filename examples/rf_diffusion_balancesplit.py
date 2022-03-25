from sklearn.ensemble import RandomForestRegressor

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
    save = 'run_rf_diffusion_balancesplit_0.3_100bins'
    points = 100

    # uq_func = poly
    # uq_coeffs_start = [0.0, 1.0, 0.1, 0.1, 0.1]
    #
    # # Load data
    # data = load_data.diffusion()
    # df = data['frame']
    # X = data['data']
    # y = data['target']
    # d = data['class_name']
    #
    # # Splitters
    # top_split = None
    # mid_split = splitters.PercentageGroupOut(n_repeats=20, groups=d, percentage=0.5)
    # bot_split = RepeatedKFold(n_splits=5, n_repeats=1)
    #
    # # ML setup
    # scale = StandardScaler()
    # selector = feature_selectors.no_selection()
    #
    # # Random for:est regression
    # grid = {}
    # model = RandomForestRegressor()
    # grid['model__n_estimators'] = [100]
    # grid['model__max_features'] = [None]
    # grid['model__max_depth'] = [None]
    # pipe = Pipeline(steps=[
    #                        ('scaler', scale),
    #                        ('select', selector),
    #                        ('model', model)
    #                        ])
    # rf = GridSearchCV(pipe, grid, cv=bot_split)
    #
    # # Evaluate
    # splits = domain.builder(
    #                         rf,
    #                         X,
    #                         y,
    #                         d,
    #                         top_split,
    #                         mid_split,
    #                         save,
    #                         seed=seed,
    #                         uq_func=uq_func,
    #                         uq_coeffs_start=uq_coeffs_start
    #                         )
    #
    # splits.assess_domain()  # Do ML
    # splits.aggregate()  # combine all of the ml data
    # statistics.folds(save)  # Gather statistics from data

    # Make parity plots
    # parity.make_plots(save, 'gpr_std')

    calibration.make_plots(save, points, 'stdcal', 'gpr_std')
    calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    calibration.make_plots(save, points, 'stdcal', 'attention_metric')
    calibration.make_plots(save, points, 'stdcal', 'attention_metric_ts_ss')
    calibration.make_plots(save, points, 'stdcal', 'oneClassSVM')


if __name__ == '__main__':
    main()
