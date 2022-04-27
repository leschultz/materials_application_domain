from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain_ctr
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration, calibration_ctr
from mad.functions import poly

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987

    p = 0.35
    id_tr = 0.15
    points = 3500 # 8000 for 0.5% OOD, 800 for 0.15% ood

    c = 3 # 9
    r = 8

    # save = f'CTR_SCT_rf_supercond_PGrOut2_{c}_clusters_{p}%ood_{points}_points_{id_tr}_id_tr_ratio'
    # save = f'CTR_EPHN_rf_supercond_PGrOut2_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_cali'
    # save = f'CTR_EPSHN_rf_supercond_PGrOut2_{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_ares'

    # save = f'CTR_EPHN_maha_rf_supercond_PGrOut2_s{p}%ood_{points}_points_{id_tr}_id_tr_ratio_err_as_cali'
    # save = f'CTR_EPHN_run_rf_supercond_cluser_{c}cluser_PGOUT_p{p}_tr{id_tr}_{points}_points_err_as_cali'

    save = f'CTR_EPHN_maha_run_rf_supercond_cluser_{c}cluser_PGOUT_p{p}_tr{id_tr}_{points}_points_err_as_cali'
    #
    uq_func = poly
    uq_coeffs_start = [0.0, 1.0, 0.1, 0.1, 0.1]
    #
    # Load data
    data = load_data.super_cond()
    df = data['frame']
    X = data['data']
    y = data['target']
    # d = data['class_name']
    #
    kmeans = KMeans(n_clusters=c, random_state=seed).fit(X)
    d = kmeans.predict(X)
    #
    # # # Splitters
    top_split = None
    # # mid_split = splitters.RepeatedClusterSplit(
    # #                                            KMeans,
    # #                                            n_repeats=r,
    # #                                            n_clusters=c
    # #                                            )
    mid_split = splitters.PercentageGroupOut2(  n_repeats=20,
                                                groups = d,
                                                percentage= p,
                                                id_tr_ratio = id_tr)

    bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

    # ML setup
    scale = StandardScaler()
    selector = feature_selectors.no_selection()
    #
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
    # #
    # # Evaluate
    splits = domain_ctr.builder(
                            rf,
                            X,
                            y,
                            d,
                            top_split,
                            mid_split,
                            save,
                            seed=seed,
                            uq_func=uq_func,
                            uq_coeffs_start=uq_coeffs_start,
                            dataset_name = 'supercond'
                            )

    splits.assess_domain()  # Do ML
    splits.aggregate()  # combine all of the ml data
    statistics.folds(save)  # Gather statistics from data

    # Make parity plots
    # parity.make_plots(save, 'gpr_std')

    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine_ctr')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'cosine')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'mahalanobis')
    #
    # calibration_ctr.make_plots(save, points, 'stdcal', 'gpr_std')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'oneClassSVM')

    # calibration.make_plots(save, points, 'stdcal', 'cosine_ctr')
    # calibration.make_plots(save, points, 'stdcal', 'cosine')
    # calibration.make_plots(save, points, 'stdcal', 'mahalanobis')
    #
    # calibration.make_plots(save, points, 'stdcal', 'gpr_std')
    # calibration.make_plots(save, points, 'stdcal', 'oneClassSVM')

    calibration_ctr.make_plots(save, points, 'stdcal', 'mahalanobis')
    calibration_ctr.make_plots(save, points, 'stdcal', 'mahalanobis_ctr')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'gpr_std')
    # calibration_ctr.make_plots(save, points, 'stdcal', 'oneClassSVM')


if __name__ == '__main__':
    main()
