import numpy as np
import polars as pl
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0


def make_simple_regression():
    x = np.array([i for i in range(100)])
    y = 2 * x + 10

    x_train, x_test, y_train, y_test = train_test_split(
        x.astype(np.int32), y.astype(np.int32), test_size=0.2, shuffle=False
    )

    train = np.concat([x_train.reshape(-1, 1), y_train.reshape(-1, 1)], axis=1)

    test = np.concat([x_test.reshape(-1, 1), y_test.reshape(-1, 1)], axis=1)

    (
        pl.DataFrame(train)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_1": "y"})
        .write_csv("data/simple_train.csv")
    )

    (
        pl.DataFrame(test)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_1": "y"})
        .write_csv("data/simple_test.csv")
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

    (
        pl.DataFrame(train)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_6": "y"})
        .write_csv("data/complex_train.csv")
    )

    (
        pl.DataFrame(test)
        .rename(lambda column_name: "x_" + column_name[-1])
        .rename({"x_6": "y"})
        .write_csv("data/complex_test.csv")
    )


if __name__ == "__main__":
    make_simple_regression()

    make_complex_regression()
