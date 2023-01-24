import sys
import os

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

sys.path.insert(0,f'{DIR}/../')

import py566

import joblib 
answers = joblib.load(str(DIR)+"/answers_Chapter1.joblib")

# Import the student solutions
import pandas as pd
import numpy as np

titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
titanic_df2 = titanic_df.loc[:,features]
titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
X = titanic_df2.drop('Cabin',axis=1)
X['CabinLetter'] = X['CabinLetter'].fillna("?")
X['Pclass'] = X['Pclass'].astype(str)
X['SibSp'] = X['SibSp'].astype(str)
X['Parch'] = X['Parch'].astype(str)
X['Age'] = ((X['Age'].fillna(X['Age'].mean())/10).astype(int)*10).astype(int).astype(str)

X = X.dropna()

X2 = X.drop(columns='Fare')
t = X['Fare']

from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.33, random_state=42)

from sklearn.metrics import mean_absolute_error
    
def test_1():
    reg = py566.base.MeanRegressor()
    reg.fit(X_train,t_train)
    results = pd.DataFrame([{'Method':'MeanRegressor','Train MAE':mean_absolute_error(t_train,reg.predict(X_train)),'Test MAE':mean_absolute_error(t_test,reg.predict(X_test))}])
    results_1 = results.set_index('Method').loc['MeanRegressor']
    answers_1 = answers.set_index('Method').loc['MeanRegressor']
    assert (results_1 == answers_1).all()
    
def test_2():
    reg = py566.tree.StumpRegressor()
    reg.fit(X_train,t_train)
    results = pd.DataFrame([{'Method':'StumpRegressor','Train MAE':mean_absolute_error(t_train,reg.predict(X_train)),'Test MAE':mean_absolute_error(t_test,reg.predict(X_test))}])
    results_1 = results.set_index('Method').loc['StumpRegressor']
    answers_1 = answers.set_index('Method').loc['StumpRegressor']
    assert (results_1 == answers_1).all()
    
def test_3():
    reg = py566.ensemble.BaggingRegressor(get_learner_func=lambda: py566.tree.StumpRegressor())
    reg.fit(X_train,t_train)
    results = pd.DataFrame([{'Method':'BaggingStumpRegressor 10','Train MAE':mean_absolute_error(t_train,reg.predict(X_train)),'Test MAE':mean_absolute_error(t_test,reg.predict(X_test))}])
    results_1 = results.set_index('Method').loc['BaggingStumpRegressor 10']
    answers_1 = answers.set_index('Method').loc['BaggingStumpRegressor 10']
    assert (results_1 == answers_1).all()