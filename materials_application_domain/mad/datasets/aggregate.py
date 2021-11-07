from pathlib import Path

import pandas as pd
import os

from mad.functions import parallel


def try_load(df):
    '''
    A function to try to load a dataset.
    '''

    try:
        return pd.read_csv(df)
    except Exception:
        pass


def folds(save, low_flag=None):
    '''
    Save aggregate data.

    inputs:
        save = The directory to save and where split data are.
    '''

    path = os.path.join(save, 'splits')

    train_paths = list(Path(save).rglob('train_split_*.csv'))
    test_paths = list(Path(save).rglob('test_split_*.csv'))

    # Load
    df_train = pd.concat(parallel(try_load, train_paths))
    df_test = pd.concat(parallel(try_load, test_paths))

    # Save data
    save = os.path.join(save, 'aggregate')
    os.makedirs(save, exist_ok=True)

    df_train.to_csv(
                    os.path.join(save, 'train_data.csv'),
                    index=False
                    )

    df_test.to_csv(
                   os.path.join(save, 'test_data.csv'),
                   index=False
                   )
