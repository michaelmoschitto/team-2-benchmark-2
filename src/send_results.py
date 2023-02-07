"""
This script holds the function we can use to send the results of a model to our server endpoint.

NOTE: EXAMPLE IS AT THE BOTTOM OF THE SCRIPT ON HOW TO FORMAT PARAMETERS!!!
"""

import requests
import time
import json


def send_results(github_username, video_game_results, life_expectancy_results):
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
    username='kroohpar_test_username2'

    video_game_model_accuracy = 0.65
    video_game_model_runtime = 13
    video_game_results = (video_game_model_accuracy, video_game_model_runtime)

    life_expectancy_model_accuracy = 0.42
    life_expectancy_model_runtime = 4
    life_expectancy_results = (life_expectancy_model_accuracy, life_expectancy_model_runtime)

    send_results(username, video_game_results, life_expectancy_results)


