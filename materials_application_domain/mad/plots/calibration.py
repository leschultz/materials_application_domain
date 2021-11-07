from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def operation(y, y_pred, op):
    '''
    Returns the desired y-axis.
    '''

    if op == 'residual':
        return np.mean(y-y_pred)
    elif op == 'rmse':
        if isinstance(y, float) and isinstance(y_pred, float):
            y = [y]
            y_pred = [y_pred]
        if len(y) == 0:
            return np.nan
        else:
            return metrics.mean_squared_error(y, y_pred)**0.5


def find_bin(df, i, sampling, points):

    if sampling == 'even':
        df['bin'] = pd.cut(
                           df[i],
                           points,
                           include_lowest=True
                           )

    elif sampling == 'equal':
        df.sort_values(by=i, inplace=True)
        df = np.array_split(df, points)
        count = 0
        for j in df:
            j['bin'] = count
            count += 1

        df = pd.concat(df)

    return df


def binner(i, data, actual, pred, save, points, sampling, ops):

    os.makedirs(save, exist_ok=True)

    name = os.path.join(save, ops)
    name += '_{}_calibration'.format(i)

    df = data[[
               i,
               actual,
               pred,
               'set',
               ]].copy()

    train = df.loc[df['set'] == 'train'].copy()
    test = df.loc[df['set'] == 'test'].copy()

    # Get bin averaging
    if (sampling is not None) and (points is not None):

        # Bin individually by set
        train = find_bin(train, i, sampling, points)  # Bin the data
        test = find_bin(test, i, sampling, points)  # Bin the data
        df = pd.concat([test, train])

        ys_train = []
        xs_train = []
        ys_test = []
        xs_test = []
        counts_train = []
        counts_test = []

        for group, values in df.groupby('bin'):

            # Compensate for empty bins
            if values.empty:
                continue

            train = values.loc[values['set'] == 'train']
            test = values.loc[values['set'] == 'test']

            x_train = train[actual].values
            y_train = train[pred].values

            x_test = test[actual].values
            y_test = test[pred].values

            y_train = operation(
                                x_train,
                                y_train,
                                ops
                                )
            x_train = np.mean(train[i].values)

            y_test = operation(
                               x_test,
                               y_test,
                               ops
                               )
            x_test = np.mean(test[i].values)

            count_train = train[i].values.shape[0]
            count_test = test[i].values.shape[0]

            ys_train.append(y_train)
            xs_train.append(x_train)

            ys_test.append(y_test)
            xs_test.append(x_test)

            counts_train.append(count_train)
            counts_test.append(count_test)

        ys_train = np.array(ys_train)
        xs_train = np.array(xs_train)

        ys_test = np.array(ys_test)
        xs_test = np.array(xs_test)

    else:

        ys_train = zip(
                       train[actual],
                       train[pred],
                       )
        ys_test = zip(
                      test[actual],
                      test[pred],
                      )

        ys_train = [operation(*i, ops) for i in ys_train]
        ys_test = [operation(*i, ops) for i in ys_test]

        xs_train = train[i].values
        xs_test = test[i].values

    xlabel = '{}'.format(i)
    xlabel = xlabel.capitalize()
    xlabel = xlabel.replace('_', ' ')

    ylabel = r'$RMSE(y, \hat{y})$'

    if (sampling is not None) and (points is not None):

        widths_train = (max(xs_train)-min(xs_train))/len(xs_train)*0.5
        widths_test = (max(xs_test)-min(xs_test))/len(xs_test)*0.5

        fig, ax = pl.subplots(2)

        ax[0].scatter(xs_test, ys_test, marker='.', color='r', label='Test')
        ax[0].scatter(xs_train, ys_train, marker='2', color='b', label='Train')

        x_ideal = [0, np.nanmax([np.nanmax(ys_test), np.nanmax(ys_train)])]
        ax[0].plot(
                   x_ideal,
                   x_ideal,
                   label='Ideal'
                   )

        ax[1].bar(
                  xs_train,
                  counts_train,
                  widths_train,
                  color='b',
                  label='Train'
                  )
        ax[1].bar(xs_test, counts_test, widths_test, color='r', label='Test')

        ax[0].set_ylabel(ylabel)

        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel('Counts')
        ax[1].set_yscale('log')

        ax[0].legend()
        ax[1].legend()

    else:
        fig, ax = pl.subplots()
        ax.scatter(xs_test, ys_test, marker='.', color='r', label='Test')
        ax.scatter(xs_train, ys_train, marker='2', color='b', label='Train')
        x_ideal = [0, np.nanmax([np.nanmax(ys_test), np.nanmax(ys_train)])]
        ax.plot(
                x_ideal,
                x_ideal,
                label='Ideal'
                )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()

    fig.tight_layout()
    fig.savefig(name)

    pl.close('all')

    data = {}
    data[ops+'_train'] = list(ys_train)
    data[xlabel+'_train'] = list(xs_train)
    data[ops+'_test'] = list(ys_test)
    data[xlabel+'_test'] = list(xs_test)

    if (sampling is not None) and (points is not None):
        data['counts_test'] = list(counts_test)
        data['counts_train'] = list(counts_train)

    jsonfile = name+'.json'
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def graphics(save, points, sampling, ops):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'splitter']
    drop_cols = groups+['pipe', 'index']

    test = pd.read_csv(os.path.join(path, 'test_data.csv'))
    train = pd.read_csv(os.path.join(path, 'train_data.csv'))

    test['set'] = 'test'
    train['set'] = 'train'

    df = pd.concat([train, test])

    # Filter for bad columns.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Do not include on x axis
    remove = {
              'y',
              'y_pred',
              'split_id',
              'flag',
              'set',
              'std',
              'std_cal',
              'features',
              'loglikelihood'
              }

    cols = ['std', 'std_cal']
    for group, values in df.groupby(groups):

        print('Plotting set {} for {}'.format(group, ops))
        values.drop(drop_cols, axis=1, inplace=True)

        group = list(map(str, group))
        parallel(
                 binner,
                 cols,
                 data=values,
                 actual='y',
                 pred='y_pred',
                 save=os.path.join(path, '_'.join(group)),
                 points=points,
                 sampling=sampling,
                 ops=ops
                 )


def make_plots(save, points=None, sampling=None):
    graphics(save, points, sampling, ops='rmse')
