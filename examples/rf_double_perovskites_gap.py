from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain, domain_ctr_cosine, domain_ctr_cosine_nn
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration
from mad.functions import poly

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run_rf_double_perovskites_gap'
    uq_func = poly
    uq_coeffs_start = [0.0, 1.0, 0.1, 0.1]

    # Load data
    data = load_data.double_perovskites_gap()
    df = data['frame']
    X = data['data']
    y = data['target']
    d = np.array( [0 for x in range( len(X))] )

    # use for NN setup
    dataset_name = 'middle'
    if X.shape[0] < 1000:
        dataset_name = 'small'
    elif X.shape[0] > 3000:
        dataset_name = 'large'

    top_split = splitters.RepeatedClusterSplit(
                       KMeans,
                       n_repeats=2,
                       n_clusters=10
                    )
    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()

    # Random forest regression
    grid = {}
    model = RandomForestRegressor()
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           ('select', selector),
                           ('model', model)
                           ])
    rf = GridSearchCV(pipe, grid, cv=bot_split)

    # Evaluate
    splits = domain_ctr_cosine_nn.builder(
                            rf,
                            X,
                            y,
                            d,
                            top_split,
                            save,
                            seed=seed,
                            uq_func=uq_func,
                            uq_coeffs_start=uq_coeffs_start,
                            dataset_name = dataset_name,
                            joint_domain = False
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data

    # Make parity plots
    parity.make_plots(save, 'gpr_std')
    calibration.make_plots(save, 'stdcal', 'gpr_std')
    parity.make_plots(save, 'cosine')
    calibration.make_plots(save, 'stdcal', 'cosine')
    parity.make_plots(save, 'cosine_ctr')
    calibration.make_plots(save, 'stdcal', 'cosine_ctr')
    parity.make_plots(save, 'cosine_nn')
    calibration.make_plots(save, 'stdcal', 'cosine_nn')


if __name__ == '__main__':
    main()
