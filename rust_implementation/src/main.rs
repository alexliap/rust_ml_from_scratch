mod linear;
use std::fs::File;
use std::io::Write;

fn fit_lin_reg_simple() {
    let results_dir = "../results/linear_regression/";

    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = results_dir.to_owned() + "data/simple_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    model.fit(&x, &y, 20000, 0.0006, false);

    // Get weight after training written to a file
    let results_path = results_dir.to_owned() + "weights/lin_reg_weights_simple_rust.txt";

    // Convert Vec<i32> to a single String
    let weights = &model
        .get_weights()
        .iter()
        .map(|n| n.to_string()) // convert each number to String
        .collect::<Vec<_>>() // collect into Vec<String>
        .join(", "); // join with comma + space

    // Write to file
    let mut file = File::create(results_path).unwrap();
    file.write_all(weights.as_bytes()).unwrap();
}

fn fit_lin_reg_complex() {
    let results_dir = "../results/linear_regression/";

    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = results_dir.to_owned() + "data/complex_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    model.fit(&x, &y, 20000, 0.001, false);

    // Get weight after training written to a file
    let results_path = results_dir.to_owned() + "weights/lin_reg_weights_complex_rust.txt";

    // Convert Vec<i32> to a single String
    let weights = &model
        .get_weights()
        .iter()
        .map(|n| n.to_string()) // convert each number to String
        .collect::<Vec<_>>() // collect into Vec<String>
        .join(", "); // join with comma + space

    // Write to file
    let mut file = File::create(results_path).unwrap();
    file.write_all(weights.as_bytes()).unwrap();
}

fn main() {
    fit_lin_reg_simple();
    fit_lin_reg_complex();
}
