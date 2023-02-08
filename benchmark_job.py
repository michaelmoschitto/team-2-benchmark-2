import sys

from tests.benchmarks import run_benchmarks
from src.main import create_model
from src.send_results import UserResults


def benchmarks():
    if create_model() is None:
        print("No model provided")
        return
    results = run_benchmarks(create_model)
    # send_results(results)
    if len(sys.argv) < 2:
        print("Error: unknown username ")
        sys.exit()
    try:
        user_results = UserResults(username=sys.argv[1], datasets=results)
    except Exception as e:
        print(f"Error: {e}")
    user_results.send_results()


if __name__ == "__main__":
    benchmarks()
