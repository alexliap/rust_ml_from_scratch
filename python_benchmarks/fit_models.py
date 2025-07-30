import numpy as np
import polars as pl
from sklearn.linear_model import SGDRegressor


def fit_lin_reg_complex():
    train = pl.read_csv("data/complex_train.csv")

    x = train.drop("y")
    y = train["y"]

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        alpha=0,
        fit_intercept=True,
        max_iter=1000,
        learning_rate="constant",
        eta0=1e-4,
    )

    model.fit(x, y)

    with open("lin_reg_weights_complex.txt", "w", encoding="utf-8") as file:
        print(
            [np.round(model.coef_, 3), np.round(model.intercept_, 3).item()], file=file
        )


def fit_lin_reg_simple():
    train = pl.read_csv("data/simple_train.csv")

    x = train.drop("y")
    y = train["y"]

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        alpha=0,
        fit_intercept=True,
        max_iter=1000,
        learning_rate="constant",
        eta0=1e-4,
    )

    model.fit(x, y)

    with open("lin_reg_weights_simple_gen.txt", "w", encoding="utf-8") as file:
        print(
            [np.round(model.coef_, 3).item(), np.round(model.intercept_, 3).item()],
            file=file,
        )


if __name__ == "__main__":
    fit_lin_reg_simple()

    fit_lin_reg_complex()
