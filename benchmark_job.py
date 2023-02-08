from tests.benchmarks import run_benchmarks
from src.main import create_model


def send_results(results: dict) -> bool:
    pass


def benchmarks():
    if create_model() is None:
        print("No model provided")
        return
    results = run_benchmarks(create_model)
    send_results(results)


if __name__ == "__main__":
    benchmarks()
