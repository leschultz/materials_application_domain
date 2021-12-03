from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def operation(y, y_pred, llh, std, stdcal, op):
    '''
    Returns the desired y-axis.
    '''

    if op == 'residual':
        return np.nanmean(y-y_pred)
    elif op == 'rmse':
        if isinstance(y, float) and isinstance(y_pred, float):
            y = [y]
            y_pred = [y_pred]
        if len(y) == 0:
            return np.nan
        else:
            return metrics.mean_squared_error(y, y_pred)**0.5
    elif op == 'llh':
        return -np.nanmean(llh)
    elif op == 'std':
        return np.nanmean(std)
    elif op == 'stdcal':
        return np.nanmean(stdcal)


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

    df = data[[
               i,
               actual,
               pred,
               'in_domain',
               'domain',
               'llh',
               'std',
               'stdcal'
               ]].copy()

    alltrain = df.loc[df['in_domain'] == True].copy()
    domains = df.loc[df['in_domain'] == False].copy()

    for group, test in domains.groupby('domain'):

        name = os.path.join(*[save, 'groups', str(group), ops])
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, i)

        # Get bin averaging
        if (sampling is not None) and (points is not None):

            # Bin individually by set
            train = find_bin(alltrain, i, sampling, points)  # Bin the data
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
                    print(values)
                    continue

                train = values.loc[values['in_domain'] == True]
                test = values.loc[values['in_domain'] == False]

                x_train = train[actual].values
                y_train = train[pred].values

                x_test = test[actual].values
                y_test = test[pred].values

                llh_train = train['llh'].values
                llh_test = test['llh'].values

                std_train = train['std'].values
                std_test = test['std'].values

                stdcal_train = train['stdcal'].values
                stdcal_test = test['stdcal'].values

                y_train = operation(
                                    x_train,
                                    y_train,
                                    llh_train,
                                    std_train,
                                    stdcal_train,
                                    ops
                                    )

                x_train = np.nanmean(train[i].values)

                y_test = operation(
                                   x_test,
                                   y_test,
                                   llh_test,
                                   std_test,
                                   stdcal_test,
                                   ops
                                   )
                x_test = np.nanmean(test[i].values)

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
                           alltrain[actual],
                           alltrain[pred],
                           alltrain['llh'],
                           alltrain['std'],
                           alltrain['stdcal']
                           )
            ys_test = zip(
                          test[actual],
                          test[pred],
                          test['llh'],
                          test['std'],
                          test['stdcal']
                          )

            ys_train = [operation(*i, ops) for i in ys_train]
            ys_test = [operation(*i, ops) for i in ys_test]

            xs_train = alltrain[i].values
            xs_test = test[i].values

        xlabel = '{}'.format(i)
        if ('logpdf' == i) or ('pdf' == i):
            xlabel = 'Negative '+xlabel

            xs_test = -1*xs_test
            xs_train = -1*xs_train
        else:
            xlabel = xlabel.capitalize()
            xlabel = xlabel.replace('_', ' ')

        if ops == 'residual':
            ylabel = r'$y-\hat{y}$'
        elif ops == 'rmse':
            ylabel = r'$RMSE(y, \hat{y})$'
        elif ops == 'llh':
            ylabel = '- Log Likelihood'
        elif ops == 'std':
            ylabel = r'$\sigma$'
        elif ops == 'stdcal':
            ylabel = r'$\sigma_{cal}$'

        if (sampling is not None) and (points is not None):

            widths_train = (max(xs_train)-min(xs_train))/len(xs_train)*0.5
            widths_test = (max(xs_test)-min(xs_test))/len(xs_test)*0.5

            fig, ax = pl.subplots(2)

            ax[0].scatter(
                          xs_test,
                          ys_test,
                          marker='.',
                          color='r',
                          label='UD'
                          )
            ax[0].scatter(
                          xs_train,
                          ys_train,
                          marker='2',
                          color='b',
                          label='IN'
                          )

            ax[1].bar(
                      xs_train,
                      counts_train,
                      widths_train,
                      color='b',
                      label='ID'
                      )
            ax[1].bar(
                      xs_test,
                      counts_test,
                      widths_test,
                      color='r',
                      label='UD'
                      )

            ax[0].set_ylabel(ylabel)

            ax[1].set_xlabel(xlabel)
            ax[1].set_ylabel('Counts')
            ax[1].set_yscale('log')

            ax[0].legend()
            ax[1].legend()

        else:
            fig, ax = pl.subplots()
            ax.scatter(xs_test, ys_test, marker='.', color='r', label='UD')
            ax.scatter(xs_train, ys_train, marker='2', color='b', label='ID')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.legend()

        fig.tight_layout()
        fig.savefig(name)

        pl.close('all')

        data = {}
        data[ops+'_id'] = list(ys_train)
        data[xlabel+'_id'] = list(xs_train)
        data[ops+'_ud'] = list(ys_test)
        data[xlabel+'_ud'] = list(xs_test)

        if (sampling is not None) and (points is not None):
            data['counts_ud'] = list(counts_test)
            data['counts_id'] = list(counts_train)

        jsonfile = name+'.json'
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)


def graphics(save, points, sampling, ops):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'splitter']
    drop_cols = groups+['pipe', 'index']

    df = pd.read_csv(os.path.join(path, 'data.csv'))

    # Filter for bad columns.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Do not include on x axis
    remove = {
              'y',
              'y_pred',
              'split_id',
              'id_count',
              'domain',
              'in_domain',
              'ud_count',
              'std',
              'stdcal',
              'features',
              'llh'
              }

    for group, values in df.groupby(groups):

        print('Plotting set {} for {}'.format(group, ops))
        values.drop(drop_cols, axis=1, inplace=True)
        cols = set(values.columns.tolist())
        cols = cols.difference(remove)

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
    graphics(save, points, sampling, ops='residual')
    graphics(save, points, sampling, ops='rmse')
    graphics(save, points, sampling, ops='llh')
    graphics(save, points, sampling, ops='stdcal')
    graphics(save, points, sampling, ops='std')
