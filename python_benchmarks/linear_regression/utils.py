import os
import pathlib

RESULTS_PATH = "linear_regression"


def make_results_path():
    current_path = pathlib.Path(__file__).parent.resolve()
    parent = current_path.parent.parent.absolute()

    resuts_dir = str(os.path.join(parent, "results", RESULTS_PATH))
    os.makedirs(resuts_dir, exist_ok=True)

    return resuts_dir
