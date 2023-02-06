import sys
import os

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

sys.path.insert(0,f'{DIR}/../')

import src

import joblib 
answers = joblib.load("tests/answers_benchmark2.joblib").set_index("Test")

# Import the student solutions
import pandas as pd
import numpy as np
    
def test_1():
    results_1 = src.benchmark_2.test_function()
    assert (results_1 == answers.loc["test 1"]["Answer"])
    

def test_2():
    results_2 = src.benchmark_2.test_function()
    assert (results_2 == answers.loc["test 2"]["Answer"])