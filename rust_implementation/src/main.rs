mod linear;

fn main() {
    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = "../data/simple_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path, target_var);

    model.fit(&x, &y, 20000, 0.0006, false);

    print!("{:?}", model.get_weights());
}
