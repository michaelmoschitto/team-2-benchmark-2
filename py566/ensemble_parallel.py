import numpy as np
import pandas as pd

from sklearn.utils import resample

from sklearn.base import BaseEstimator, RegressorMixin

import base

# from timer import timer_func

import ray


    # ------------------------------------
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
  






#

def get_learner_example():
    return base.MeanRegressor()

def boostrap_sample(X,y):
    X, y = resample(X, y)
    return X,y

@ray.remote
def fit_one_tree(X, y, seed):

    np.random.seed(seed)
    model = StumpRegressor()
    X_train, y_train = boostrap_sample(X, y)
    model = model.fit(X_train, y_train)
    return model

ensemble_trees = []

def fit( X, y):



    trained_trees = [fit_one_tree.remote(X, y, seed =42) for i in range(5)]
    ensemble_trees = ray.get(trained_trees)
    
    return ensemble_trees


def predict(X, trees):
        # the code below can be modified, but I leave it here as a clue to my implementation
        tree_predictions = []
        for j in range(len(trees)):
            tree = trees[j]
            tree_predictions.append(tree.predict(X).tolist())
        return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

        # return [100] * len(X)

# class BaggingRegressor(BaseEstimator, RegressorMixin):
#     def __init__(self, ntrees=10, get_learner_func=get_learner_example,seed=42):
#         self._get_learner_func = get_learner_func
#         self._ntrees = ntrees
#         self._seed = seed
#         self._trees = []


import pandas as pd
from sklearn.metrics import mean_absolute_error

X_train = pd.read_csv("Lab1_Data/X_train.csv")
X_test = pd.read_csv("Lab1_Data/X_test.csv")
t_train = pd.read_csv("Lab1_Data/t_train.csv")
t_test = pd.read_csv("Lab1_Data/t_test.csv")

results = []

trained_trees = fit(X_train,t_train)

print(trained_trees)

y_pred = predict(X_test, trained_trees)
print(y_pred)
# results.append({'Method':'BaggingStumpRegressor 100',
#                 'Train MAE':mean_absolute_error(t_train,predict(X_train, trained_trees)),
#                 'Test MAE':mean_absolute_error(t_test, predict(X_test, trained_trees))})
# pd.DataFrame(results)




    # @timer_func
















