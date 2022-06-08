from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.ml import splitters, feature_selectors, domain, domain_ctr_cosine
from mad.datasets import load_data, statistics
from mad.plots import parity, calibration
from mad.functions import poly

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = './../agg_exp/supercond'

    aggregation = True
    # Make parity plots
    # parity.make_plots(save, 'gpr_std', aggregation)
    calibration.make_plots(save, 'stdcal', 'gpr_std', aggregation)
    # parity.make_plots(save, 'mahalanobis')
    # calibration.make_plots(save, 'stdcal', 'mahalanobis')
    # parity.make_plots(save, 'cosine_ctr', aggregation)
    calibration.make_plots(save, 'stdcal', 'cosine_ctr', aggregation)


if __name__ == '__main__':
    main()
