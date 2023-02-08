"""
This script holds the function we can use to send the results of a model to our server endpoint.

NOTE: EXAMPLE IS AT THE BOTTOM OF THE SCRIPT ON HOW TO FORMAT PARAMETERS!!!
"""
import sys

import requests
import time
import json

from dataclasses import dataclass, asdict

@dataclass
class UserResults:
    username: str
    datasets: dict
    timestamp: float

    def __init__(self, username: str = "unknown_user", datasets: dict = {}):
        self.username = username
        self.datasets = datasets
        self.timestamp = 0.
        return

    def add_dataset(self, dataset_name: str, metrics: dict):
        """ adds a dataset + metrics to user """
        self.datasets[dataset_name] = metrics
        return

    def add_metric_to_dataset(self, dataset_name: str, metric_name: str, value: float):
        """ adds a metric to an existing dataset """
        self.datasets[dataset_name][metric_name] = value
        return
    
    def get_metrics(self):
        """ returns a list of metrics that are used in the datasets """
        metrics = []
        for dataset in self.datasets.keys():
            for metric in list(self.datasets[dataset].keys()):
                metrics.append(metric)
        return list(set(metrics))

    def send_results(self):
        """ Sends a request containing user submission to firebase db """
        ENDPOINT_URL = 'https://csc-566-benchmarks-default-rtdb.firebaseio.com/benchmark-2.json'

        self.timestamp = time.time()
        resp = requests.post(ENDPOINT_URL, json=asdict(self))

        if resp.status_code != 200:
            print(f"Error {resp.status_code}: Issue with post request to backend")
            return False
        return True

def old_send_results(github_username, video_game_results, life_expectancy_results):
    """
    This function sends our results to our database on Firebase. 

    Args:
        github_username (str): The string github username
        video_game_results (tuple): This is a tuple of the results of the model on video game 
            data. It is in the format:
                -> (Model Accuracy: float, Model Runtime: Int)
        life_expectancy_results (tuple): This is a tuple of the results of the model on video game 
            data. It is in the format:
                -> (Model Accuracy: float, Model Runtime: Int)
            
    """
    ENDPOINT_URL = 'https://csc-566-benchmarks-default-rtdb.firebaseio.com/benchmark-2.json'

    # Parse results from tuple parameters
    vg_acc, vg_time = video_game_results
    le_acc, le_time = life_expectancy_results

    # Create object for post request
    obj = {}
    datasets_obj = {}

    obj["username"] = github_username
    obj["timestamp"] = time.time()
    obj["datasets"] = {}

    datasets_obj["video_games"] = {"accuracy": vg_acc, "runtime": vg_time}
    datasets_obj["life_expectancy"] = {"accuracy": le_acc, "runtime": le_time}

    obj["datasets"] = datasets_obj

    # print(f'obj: {obj}')
    resp = requests.post(ENDPOINT_URL, json=obj)

    if resp.status_code != 200:
        print("Error - Issue with post request to backend :(")

if __name__ == '__main__':
    username='dummy_username_1'

    video_game_model_accuracy = 0.33
    video_game_model_runtime = 15
    # video_game_results = (video_game_model_accuracy, video_game_model_runtime)
    video_game_metrics = {"accuracy": video_game_model_accuracy, "runtime": video_game_model_runtime}

    life_expectancy_model_accuracy = 0.76
    life_expectancy_model_runtime = 8
    # life_expectancy_results = (life_expectancy_model_accuracy, life_expectancy_model_runtime)
    life_expectancy_metrics = {"accuracy": life_expectancy_model_accuracy, "runtime": life_expectancy_model_runtime}

    # old_send_results(username, video_game_results, life_expectancy_results)
    user_results = UserResults(
        username=username,
        datasets={
            "Video Games": video_game_metrics
        }
    )
    user_results.add_dataset("Life Expectancy", life_expectancy_metrics)
    user_results.send_results()

    sys.exit(0)


