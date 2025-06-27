import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression


def fit_lin_reg():
    train = pl.read_csv("data/complex_train.csv")

    x = train.drop("y")
    y = train.select("y")

    model = LinearRegression()

    model.fit(x, y)

    with open("lin_reg_weights_complex.txt", "w", encoding="utf-8") as file:
        print(np.round(model.coef_, 3), file=file)


if __name__ == "__main__":
    fit_lin_reg()
