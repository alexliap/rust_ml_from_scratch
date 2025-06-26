import numpy as np
import polars as pl
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0


def make_simple_regression():
    pass


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

    train = {"x": x_train, "y": y_train}

    test = {"x": x_test, "y": y_test}

    pl.DataFrame(train).write_csv("train.csv")
    pl.DataFrame(test).write_csv("test.csv")


if __name__ == "__main__":
    make_complex_regression()
