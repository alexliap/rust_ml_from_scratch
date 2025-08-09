import os

import numpy as np
import polars as pl
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from utils import make_results_path


def get_test_results(model: SGDRegressor, x_test: pl.DataFrame, y_true: pl.Series):
    y_preds = model.predict(X=x_test)

    loss = mean_squared_error(y_true=y_true, y_pred=y_preds)

    return loss


def fit_lin_reg_complex():
    results_dir = make_results_path()

    train = pl.read_csv(os.path.join(results_dir, "data/complex_train.csv"))

    x = train.drop("y")
    y = train["y"]

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        alpha=0,
        fit_intercept=True,
        max_iter=20000,
        learning_rate="constant",
        eta0=1e-4,
    )

    model.fit(x, y)

    weight_dir = os.path.join(results_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)

    weights_file = os.path.join(weight_dir, "lin_reg_weights_complex.txt")

    with open(weights_file, "w", encoding="utf-8") as file:
        weights = np.round(model.coef_, 3).tolist() + [
            np.round(model.intercept_, 3).item()
        ]
        weights = ", ".join([str(weight) for weight in weights])
        print(weights, file=file)

    test = pl.read_csv(os.path.join(results_dir, "data/complex_test.csv"))

    x = test.drop("y")
    y = test["y"]

    loss = get_test_results(model, x_test=x, y_true=y)

    return loss


def fit_lin_reg_simple():
    results_dir = make_results_path()

    train = pl.read_csv(os.path.join(results_dir, "data/simple_train.csv"))

    x = train.drop("y")
    y = train["y"]

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        alpha=0,
        fit_intercept=True,
        max_iter=20000,
        learning_rate="constant",
        eta0=1e-4,
    )

    model.fit(x, y)

    weight_dir = os.path.join(results_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)

    weights_file = os.path.join(weight_dir, "lin_reg_weights_simple.txt")

    with open(weights_file, "w", encoding="utf-8") as file:
        weights = np.round(model.coef_, 3).tolist() + [
            np.round(model.intercept_, 3).item()
        ]
        weights = ", ".join([str(weight) for weight in weights])
        print(weights, file=file)

    test = pl.read_csv(os.path.join(results_dir, "data/simple_test.csv"))

    x = test.drop("y")
    y = test["y"]

    loss = get_test_results(model, x_test=x, y_true=y)

    return loss


if __name__ == "__main__":
    results_dir = make_results_path()

    simple_loss = fit_lin_reg_simple()

    complex_loss = fit_lin_reg_complex()

    loss_results_path = os.path.join(results_dir, "loss.csv")

    pl.DataFrame(
        {"Python": [simple_loss, complex_loss], "Rust Implementation": [None, None]}
    ).write_csv(loss_results_path)
