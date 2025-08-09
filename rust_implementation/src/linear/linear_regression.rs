use ndarray::{Array, OwnedRepr};
use ndarray::{Axis, concatenate};
use polars::prelude::*;

pub struct LinearRegression {
    weights: Vec<f32>,
}

impl LinearRegression {
    pub fn new() -> Self {
        return Self {
            weights: Vec::new(),
        };
    }

    pub fn fit(
        &mut self,
        x: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
        y: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
        epochs: i32,
        lr: f32,
        verbose: bool,
    ) {
        self._init_weights(x.shape()[1] as i32);

        let mut array_weights = Array::from_vec(self.weights.clone())
            .into_shape_clone((self.weights.len(), 1))
            .unwrap();

        for epoch in 1..=epochs {
            let grad = self._grad(x, y);

            array_weights = array_weights - lr * grad.clone();

            self.weights = array_weights.flatten().to_vec();

            let loss = self.loss(x, y);

            if verbose {
                print!(
                    "\nEpoch: {:?} | Loss: {:?} | Weights: {:?}\n",
                    epoch,
                    loss,
                    array_weights.flatten().to_vec(),
                )
            }
        }
    }

    fn _init_weights(&mut self, num_vars: i32) {
        for _ in 1..=num_vars {
            self.weights.push(0 as f32)
        }
        // bias weight
        self.weights.push(0 as f32)
    }

    fn _grad(
        &self,
        x: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
        y: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
    ) -> ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
        let curr_preds = self.predict(x);

        let cast_x = x.mapv(|x| x as f32);
        let cast_y = y.mapv(|y| y as f32);

        let x_bias = Array::from_elem((cast_x.shape()[0], 1), 1.);

        let x_data = concatenate![Axis(1), cast_x, x_bias];

        let diff = &(curr_preds - cast_y);
        let agg = x_data.t().dot(diff);

        let grad = (1. / (x.shape()[0] as f32)) * agg;

        return grad;
    }

    pub fn loss(
        &self,
        x: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
        y: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
    ) -> f32 {
        let cast_y = y.mapv(|y| y as f32);
        let error = self.predict(x) - cast_y;
        let mean_squared_error = error.mapv(|error| error.powi(2)).mean().unwrap();

        return mean_squared_error;
    }

    pub fn predict(
        &self,
        x: &ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
        let array_weights = Array::from_vec(self.weights.clone())
            .into_shape_clone((self.weights.len(), 1))
            .unwrap();

        let cast_x = x.mapv(|x| x as f32);

        let x_bias = Array::from_elem((cast_x.shape()[0], 1), 1.);

        let x_data = concatenate![Axis(1), cast_x, x_bias];

        let preds = x_data.dot(&array_weights);

        return preds;
    }

    pub fn load_data(
        &self,
        path: &str,
        target: &str,
    ) -> (
        ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
        ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>,
    ) {
        let file = std::fs::File::open(path).unwrap();

        let df = CsvReader::new(file).finish().unwrap();

        let x = df
            .drop(target)
            .unwrap()
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();

        let y = df
            .select([target])
            .unwrap()
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();

        return (x, y);
    }

    pub fn get_weights(self) -> Vec<f32> {
        return self.weights.clone();
    }
}
