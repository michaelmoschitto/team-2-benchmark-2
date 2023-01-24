import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from . import base

def gain_ratio(y,x):
    y_mean = base.MeanRegressor().fit(x.to_frame(),y).predict(x.to_frame())
    base_mae = mean_absolute_error(y,y_mean)
    after_split_weight_mae = 0
    for f in x.unique():
        mask = x == f
        ysplit = y.loc[mask]
        xsplit = x.loc[mask]
        split_y_mean = base.MeanRegressor().fit(xsplit.to_frame(),ysplit).predict(xsplit.to_frame())
        split_mae = mean_absolute_error(ysplit,split_y_mean)
        after_split_weight_mae += len(ysplit)/len(y)*split_mae
    return (base_mae - after_split_weight_mae)/base_mae

class StumpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, sample_param_here="Not using this"):
        self.sample_param_here = sample_param_here

        # column with highest gain ratio
        self._split_column = None
        
        # dictionary of row_val: prediction
        # row_val: a unique value in the "best" column identified by gain_ratio()
        # prediction: the mean of all y's that have that row_val
        self._predictions = {}

        # used if val in X_test does not have stump
        self._mean = 0
            
    def fit(self, X, y):

        # create all the stumps
        self._split_column = sorted(X.columns, key=lambda col: gain_ratio(y, X[col]), reverse=True)[0]

        # self._mean = np.mean(y[X.index[X[self._split_column]].tolist()])
        self._mean = np.mean(y)

        for val in X[self._split_column].unique():
            indices = X.index[X[self._split_column] == val].tolist()
            stump = np.mean(y[indices])
            self._predictions[val] = stump
        return self
    
    def predict(self, X):
        # the code below can be modified, but I leave it here as` a clue to my implementation
        predictions = []
        for value in X[self._split_column]:
            if value in self._predictions:
                predictions.append(self._predictions[value])
            else:
                predictions.append(self._mean)
            
        return np.array(predictions)