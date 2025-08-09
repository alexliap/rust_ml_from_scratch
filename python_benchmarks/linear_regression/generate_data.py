import os

import numpy as np
import polars as pl
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from utils import make_results_path

RANDOM_STATE = 0


def make_simple_regression():
    x = np.random.uniform(0, 100, 500)
    y = 2 * x + 10

    x_train, x_test, y_train, y_test = train_test_split(
        x.astype(np.int32), y.astype(np.int32), test_size=0.2, shuffle=False
    )

    train = np.concat([x_train.reshape(-1, 1), y_train.reshape(-1, 1)], axis=1)

    test = np.concat([x_test.reshape(-1, 1), y_test.reshape(-1, 1)], axis=1)

    results_dir = make_results_path()
    data_dir = os.path.join(results_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "simple_train.csv")
    test_path = os.path.join(data_dir, "simple_test.csv")

    (
        pl.DataFrame(train)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_1": "y"})
        .write_csv(train_path)
    )

    (
        pl.DataFrame(test)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_1": "y"})
        .write_csv(test_path)
    )


def make_complex_regression():
    x, y = make_regression(
        n_samples=1000,
        n_features=6,
        n_informative=3,
        n_targets=1,
        random_state=RANDOM_STATE,
        coef=False,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x.astype(np.float32),
        y.astype(np.float32),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train = np.concat([x_train, y_train.reshape(-1, 1)], axis=1)

    test = np.concat([x_test, y_test.reshape(-1, 1)], axis=1)

    results_dir = make_results_path()
    data_dir = os.path.join(results_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "complex_train.csv")
    test_path = os.path.join(data_dir, "complex_test.csv")

    (
        pl.DataFrame(train)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_6": "y"})
        .write_csv(train_path)
    )

    (
        pl.DataFrame(test)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_6": "y"})
        .write_csv(test_path)
    )


if __name__ == "__main__":
    make_simple_regression()

    make_complex_regression()
