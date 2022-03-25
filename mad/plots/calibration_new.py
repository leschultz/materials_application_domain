from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as pl
from mad.functions import chunck
from sklearn import metrics

import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import json
import os


def make_plots(save, bin_size, xaxis, dist, thresh=0.1):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    df['ares'] = abs(df['y'].values-df['y_pred'].values)
    df = df.sort_values(by=[xaxis, 'ares', dist])

    if (dist == 'pdf') or (dist == 'logpdf'):
        sign = -1.0
        dist_label = 'negative '+dist
    else:
        sign = 1.0
        dist_label = dist

    maxx = []
    maxy = []
    minx = []
    miny = []
    vmin = []
    vmax = []
    xs = []
    ys = []
    cs = []
    ds = []
    zs = []

    rows = []
    rows_table = []
    for subgroup, subvalues in df.groupby('in_domain'):

        if subgroup == 'td':
            continue

        x = subvalues[xaxis].values
        y = subvalues['ares'].values
        c = subvalues[dist].values*sign

        '''
        x = list(chunck(x, bin_size))
        y = list(chunck(y, bin_size))
        c = list(chunck(c, bin_size))

        # Skip values that are empty
        if (not x) or (not y) or (not c):
            continue

        # Mask values
        x = np.ma.masked_invalid(x)
        y = np.ma.masked_invalid(y)
        c = np.ma.masked_invalid(c)

        x = np.array([np.ma.mean(i) for i in x])
        y = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in y])
        c = np.array([np.ma.mean(i) for i in c])
        '''

        y **= 2

        # Normalization
        x = x/std
        y = y/std

        z = abs(y-x)

        # Table data
        rmse = metrics.mean_squared_error(y, x)**0.5
        r2 = metrics.r2_score(y, x)

        if subgroup == 'ud':
            domain_name = 'LO-Cluster'
        elif subgroup == 'id':
            domain_name = 'LO-Random'
        elif subgroup == 'td':
            domain_name = 'Train LO-Random'
        else:
            domain_name = 'Error'

        rows.append([domain_name, rmse, r2])

        domain_name = '{}'.format(domain_name)
        rmse = '{:.2E}'.format(rmse)
        r2 = '{:.2f}'.format(r2)

        rows_table.append([domain_name, rmse, r2])

        maxx.append(np.ma.max(x))
        maxy.append(np.ma.max(y))
        minx.append(np.ma.min(x))
        miny.append(np.ma.min(y))
        vmin.append(np.ma.min(c))
        vmax.append(np.ma.max(c))
        xs.append(x)
        ys.append(y)
        cs.append(c)
        zs.append(z)
        ds.append(subgroup)

    minx = np.append(minx, 0.0)
    miny = np.append(miny, 0.0)

    vmin = np.ma.min(vmin)
    vmax = np.ma.max(vmax)
    maxx = np.ma.max(maxx)
    maxy = np.ma.max(maxy)
    minx = np.ma.min(minx)
    miny = np.ma.min(miny)

    # For plot data export
    data_cal = {}
    data_err = {}
    data_rmse = {}

    err_y_label = r'|RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$|'

    fig, ax = pl.subplots()
    fig_err, ax_err = pl.subplots()
    fig_roc, ax_roc = pl.subplots()
    fig_pr, ax_pr = pl.subplots()
    fig_rmse, ax_rmse = pl.subplots()
    for x, y, c, z, subgroup in zip(xs, ys, cs, zs, ds):

        if subgroup == 'id':
            domain = 'LO-Random'
            marker = '1'
            zorder = 3
        elif subgroup == 'ud':
            domain = 'LO-Cluster'
            marker = 'x'
            zorder = 2
        elif subgroup == 'td':
            domain = 'Train LO-Random'
            marker = '.'
            zorder = 1
        else:
            marker = '*'
            zorder = 0

        dens = ax.scatter(
                          x,
                          y,
                          c=c,
                          marker=marker,
                          label='{}'.format(domain),
                          cmap=pl.get_cmap('viridis'),
                          vmin=vmin,
                          vmax=vmax,
                          zorder=zorder
                          )

        ax_err.scatter(
                       c,
                       z,
                       marker=marker,
                       label='{}'.format(domain),
                       zorder=zorder
                       )

        ax_rmse.scatter(
                        c,
                        y,
                        marker=marker,
                        label='{}'.format(domain),
                        zorder=zorder
                        )

        data_err[domain] = {}
        data_err[domain][dist_label] = c.tolist()
        data_err[domain][err_y_label] = z.tolist()

        data_cal[domain] = {}
        data_cal[domain][r'$\sigma_{m}/\sigma_{y}$'] = x.tolist()
        data_cal[domain][r'RMSE/$\sigma_{y}$'] = y.tolist()
        data_cal[domain][dist_label] = c.tolist()

        data_rmse[domain] = {}
        data_rmse[domain][r'RMSE/$\sigma_{y}$'] = y.tolist()
        data_rmse[domain][dist_label] = c.tolist()

    ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

    ax.set_xlim([minx-0.1*abs(minx), maxx+0.1*abs(maxx)])
    ax.set_ylim([miny-0.1*abs(minx), maxy+0.1*abs(maxx)])

    ax.legend()
    ax.set_xlabel(r'$\sigma_{m}/\sigma_{y}$')
    ax.set_ylabel(r'RMSE/$\sigma_{y}$')

    ax_err.axhline(thresh, color='r', linestyle=':', label='Boundary')
    ax_err.legend()
    ax_err.set_xlabel(dist_label)
    ax_err.set_ylabel(err_y_label)

    ax_rmse.axhline(thresh, color='r', linestyle=':', label='Boundary')
    ax_rmse.legend()
    ax_rmse.set_xlabel(dist_label)
    ax_rmse.set_ylabel(r'RMSE/$\sigma_{y}$')

    cbar = fig.colorbar(dens)
    cbar.set_label(dist_label)

    # Make a table
    cols = [r'Domain', r'RMSE', r'$R^2$']
    table = ax.table(
                     cellText=rows_table,
                     colLabels=cols,
                     colWidths=[0.22]*2+[0.1],
                     loc='lower right',
                     )

    data_cal['table'] = {}
    data_cal['table']['metrics'] = cols
    data_cal['table']['rows'] = rows

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.25, 1.25)

    fig.tight_layout()
    fig_err.tight_layout()
    fig_rmse.tight_layout()

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'calibration',
            xaxis+'_vs_'+dist
            ]

    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'calibration.png')
    fig.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_cal, handle)

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'err_in_err',
            xaxis+'_vs_'+dist
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'err_in_err.png')
    fig_err.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_err, handle)

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'rmse',
            xaxis+'_vs_'+dist
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'rmse.png')
    fig_rmse.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_rmse, handle)
