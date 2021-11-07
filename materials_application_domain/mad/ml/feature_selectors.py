from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

class no_selection:
    '''
    A class used to skip feature selection.
    '''

    def transform(self, X):
        '''
        Select all columns
        '''

        return X

    def fit(self, X, y=None):
        return self
    
class vif_selection:
    '''
    A class use vif score for feature selection. 
    '''
    def __init__(self, percent = 0.8):
        '''
        percent: gives the top percent of feature
        '''
        self.percent = percent
    
    def transform(self, X):
        '''
        Find the top percent of features, and transform the data
        '''
        feature_indices = [ index for index in range(X.shape[1]) ]

        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in feature_indices ]
        vif["features_index"] = feature_indices
        vif_sorted = vif.sort_values("VIF Factor")

        num_to_select = int( len(feature_indices) * self.percent )
        feature_selected = list( vif_sorted["features_index"] )[: num_to_select]

        X = X[:,feature_selected]

        return X
    
    def fit(self, X, y=None):
        return self
