from matplotlib import pyplot as pl
from itertools import combinations

import seaborn as sns
import pandas as pd
import os

from mad.functions import parallel


def plot(cols, df, save):
    cols = list(cols)
    data = df[cols]

    fig = sns.pairplot(data, kind="kde")
    fig.tight_layout()
    fig.savefig(os.path.join(save, '_'.join(cols)))


def make_plots(df, save):

    # Output directory creation
    save = os.path.join(save, 'kde')
    os.makedirs(save, exist_ok=True)
    parallel(plot, list(combinations(df.columns, 2)), df=df, save=save)
