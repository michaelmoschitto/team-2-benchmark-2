# Team-2 Benchmark-2 Student Repo

## Repository Structure

- /data: folder containg datasets used for benchmark tasks
- /notebooks: notebooks to run and test code written in /src
- /src: source code that will be autograded in GitHub

## Grading

Unit tests will be run on code contained in /src folder. Nothing in /notebooks will be graded

## Submission Process

1. In `src/main.py`, fill out the `create_model` function. This function should initialize and return the model that you want to use in the benchmarks.
2. Ensure that your model runs before submitting. Run `python benchmark_job.py` to double check that your model works.
3. Push your code to submit. The benchmarks will run in a GitHub action. You can submit as many times as you want. Your latest submission will appear on the leaderboard.
4. After a few minutes, check the [leaderboard site](https://csc-566-benchmark-results.netlify.app/) to see your results.

## Setup a virtual env

From the command line in the root directory run:

1. `python3.9 -m venv .venv`
2. `source .venv/bin/activate`
3. ` pip install -r requirements.txt`
