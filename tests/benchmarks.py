import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from benchmark_utils import (
    preprocess_flare_df,
    preprocess_life_df,
    preprocess_video_games_df,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import timeit


def run_benchmarks(model_func) -> dict:
    SEED = 42
    TEST_SIZE = 0.2
    TIMING_ITERATIONS = 3

    datasets = {
        "video_games": preprocess_video_games_df("data/video_games.csv"),
        "life": preprocess_life_df("data/life_expectancy.csv"),
        "flare": preprocess_flare_df(
            link="https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2",
            sep=" ",
            names=[
                "class",
                "size",
                "spot",
                "activity",
                "evolution",
                "prev24hr",
                "histcomplex",
                "histcomplexsundisk",
                "area",
                "largest-spotarea",
                "C",
            ],
        ),
    }

    np.random.seed(SEED)
    results = {}

    for dataset_name, (X, y) in datasets.items():
        n_iterations = 1 if dataset_name == "video_games" else TIMING_ITERATIONS
        model = model_func()
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )
        preds = None

        def benchmark():
            global preds
            model.fit(x_train, y_train)
            preds = model.predict(x_test)

        perf = timeit.timeit(benchmark, number=n_iterations) / n_iterations
        results[dataset_name] = {
            "mse": mean_squared_error(y_test, preds),
            "runtime": perf,
        }
