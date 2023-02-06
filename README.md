# Team-2 Benchmark-2 Student Repo 


## Repository Structure
* /data: folder containg datasets used for benchmark tasks
* /notebooks: notebooks to run and test code written in /src
* /src: source code that will be autograded in GitHub


## Grading
Unit tests will be run on code contained in /src folder. Nothing in /notebooks will be graded

## Run Tests Locally
From the command line and in the tests folder, it is recommended you run:

``
pytest -vv --diff-symbols
``

## Setup a virtual env

From the command line in the root directory run: 
1. ``python3.9 -m venv .venv``
2. ``source .venv/bin/activate``
3. `` pip install -r requirements.txt``

