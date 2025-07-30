mod linear;

fn main() {
    let mut model = linear::linear_regression::LinearRegression::new();

    let data_path = "../data/simple_train.csv";
    let target_var = "y";

    let (x, y) = model.load_data(data_path, target_var);

    // print!("x: {} | y: {}", x, y);

    // print!("x.shape: {:?} | y.shape: {:?}", x.shape()[0], y.shape()[1]);

    // model.fit(&x, &y);

    // for row in x {
    //     print!("{row}")
    // }

    // let preds = model.predict(&x);

    // print!("{preds}")
    model.fit(&x, &y, 40, 0.0001);
}
