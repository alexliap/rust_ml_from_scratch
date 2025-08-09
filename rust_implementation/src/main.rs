mod linear;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;

fn fit_lin_reg_simple(epochs: i32, lr: f32) -> f32 {
    let results_dir = "../results/linear_regression/";

    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = results_dir.to_owned() + "data/simple_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    model.fit(&x, &y, epochs, lr, false);

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

    // Get loss on test data
    let data_path = results_dir.to_owned() + "data/simple_test.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    let loss = model.loss(&x, &y);

    return loss;
}

fn fit_lin_reg_complex(epochs: i32, lr: f32) -> f32 {
    let results_dir = "../results/linear_regression/";

    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = results_dir.to_owned() + "data/complex_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    model.fit(&x, &y, epochs, lr, false);

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

    // Get loss on test data
    let data_path = results_dir.to_owned() + "data/complex_test.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path.as_str(), target_var);

    let loss = model.loss(&x, &y);

    return loss;
}

fn main() {
    let results_dir = "../results/linear_regression/";
    let loss_data_path = results_dir.to_owned() + "loss.csv";

    let simple_loss = fit_lin_reg_simple(20000, 0.0005);
    let complex_loss = fit_lin_reg_complex(10000, 0.0003);

    print!(
        "Simple Loss: {:?} | Complex Loss: {:?}",
        simple_loss, complex_loss
    );

    let loss_vec = Vec::<f32>::from([simple_loss, complex_loss]);
    let col = Series::new("Rust Implementation".into(), loss_vec);

    let file = File::open(&loss_data_path).unwrap();
    let mut loss_df = CsvReader::new(file).finish().unwrap();
    // Replace the column
    let loss_df = loss_df.with_column(col).unwrap();

    // Overwrite the same file
    let mut file = File::create(&loss_data_path).unwrap();
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(loss_df)
        .unwrap();
}
