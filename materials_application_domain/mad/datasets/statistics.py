from sklearn import metrics
from pathlib import Path

import pandas as pd
import numpy as np
import os

from mad.functions import parallel


def eval_reg_metrics(groups):
    '''
    Evaluate standard regression prediction metrics.
    '''

    group, df = groups

    y = df['y']
    y_pred = df['y_pred']

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sig = rmse/np.std(y)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    model, scaler, splitter, split_id, flag = group

    results = {}
    results['model'] = model
    results['scaler'] = scaler
    results['splitter'] = splitter
    results['split_id'] = split_id
    results[r'$RMSE$'] = rmse
    results[r'$RMSE/\sigma$'] = rmse_sig
    results[r'$MAE$'] = mae
    results[r'$R^{2}$'] = r2
    results['flag'] = flag

    return results


def group_metrics(df, cols):
    '''
    Get the metrics statistics.
    '''

    groups = df.groupby(cols)
    mets = parallel(eval_reg_metrics, groups)
    mets = pd.DataFrame(mets)

    return mets


def stats(df, cols):
    '''
    Get the statistic of a dataframe.
    '''

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)
    df = df.reset_index()

    return df


def folds_opperation(save, file_name, low_flag):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
        low_flag = Flagg all values with less than this logpdf.
    '''

    save = os.path.join(save, 'aggregate')

    # Load
    df_path = os.path.join(save, file_name)
    df = pd.read_csv(df_path)
    dfstats = stats(df, ['index', 'model', 'scaler', 'splitter'])

    if isinstance(low_flag, float) & ('logpdf_mean' in dfstats.columns):
        dfstats['flag'] = dfstats['logpdf_mean'] <= low_flag

        if 'flag' in df.columns:
            df.drop('flag', axis=1, inplace=True)

        df = dfstats[['flag', 'index']].merge(df, on='index')
    else:
        dfstats['flag'] = False
        df['flag'] = False

    mets = group_metrics(df, [
                              'model',
                              'scaler',
                              'splitter',
                              'split_id',
                              'flag'
                              ])

    # Get statistics
    metsstats = mets.drop('split_id', axis=1)
    metsstats = stats(metsstats, [
                                  'model',
                                  'scaler',
                                  'splitter',
                                  'flag'
                                  ])

    # Save data
    df.to_csv(df_path, index=False)
    set_type = file_name.split('_')[0]
    dfstats.to_csv(
                   os.path.join(save, set_type+'_data_stats.csv'),
                   index=False
                   )
    mets.to_csv(
                os.path.join(save, set_type+'_metrics.csv'),
                index=False
                )
    metsstats.to_csv(
                     os.path.join(save, set_type+'_metrics_stats.csv'),
                     index=False
                     )


def folds(save, low_flag=None):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
        low_flag = Flagg all values with less than this logpdf.
    '''

    folds_opperation(save, 'test_data.csv', low_flag)
    folds_opperation(save, 'train_data.csv', low_flag)
