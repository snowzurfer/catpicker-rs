extern crate nalgebra as na;

use std::io;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::ops::{SubAssign};
use std::time::Instant;
use na::{DVector, DMatrix, Real};
use std::iter::IntoIterator;

fn load_cifar(fname: &str) -> io::Result<(DMatrix<f32>, DVector<f32>)> {
    let f = File::open(fname)?;

    // We hardcode these values because we know beforehand the dimensionalities
    // of the CIFAR-10 training set
    let cfar_m = 10000usize;
    let cfar_n = 3072usize;

    let mut x_mx = DMatrix::<f32>::zeros(cfar_n, cfar_m);
    let mut y_mx = DVector::<f32>::zeros(cfar_m);

    let mut reader = BufReader::new(f);
    let mut raw_data = Vec::new();

    reader.read_to_end(&mut raw_data)?;
    for i in 0..cfar_m {
        // Read first byte which tells us the category of the image
        let label = raw_data[(cfar_n + 1) * i];

        // If cat, set it as one
        if label == 3 {
            y_mx[i] = 1f32;
        }
        // If not, set it as zero
        else {
            y_mx[i] = 0f32;
        }

        // Read the data of an image into a column of the data matrix
        for (ix, f) in x_mx.column_mut(i).iter_mut().take(cfar_n).enumerate() {
            *f= raw_data[(ix + 1) + ((cfar_n + 1) * i)] as f32;
        }
    }

    Ok((x_mx, y_mx))
}

fn standardise_rgb<T: Real>(x_mx: &DMatrix<T>, div: T) -> DMatrix<T>
{
    x_mx / div
}

fn sigmoid<'a, I: IntoIterator<Item=&'a mut f32>>(it: I) {
    for v in it {
        *v = 1f32 / (1f32  + (-(*v)).exp());
    }
}

fn ln_v<T: Real>(x_mx: &DVector<T>) -> DVector<T> {
    DVector::from_iterator(x_mx.shape().0,
                           x_mx.iter().map(|v| v.ln()))
}

fn propagate(w: &DVector<f32>, b: f32, x_mx: &DMatrix<f32>,
             y_mx: &DVector<f32>) -> ((DVector<f32>, f32), f32) {
    // Compute activation value
    let mut a_mx = (w.transpose() * x_mx).add_scalar(b).transpose();
    sigmoid(&mut a_mx.iter_mut());

    // Compute cost
    let cost = -((y_mx.component_mul(&ln_v(&a_mx)) +
                 ((-y_mx).add_scalar(1f32))
                    .component_mul(&ln_v(&(-&a_mx).add_scalar(1f32))))
                .iter().sum::<f32>()
                / x_mx.shape().1 as f32);

    // Do backprop to find dJ/dW and dJ/db
    let diff = a_mx - y_mx;
    let dw = (x_mx * &diff) / x_mx.shape().1 as f32;
    let db = diff.into_iter().sum::<f32>() / x_mx.shape().1 as f32;

    ((dw, db), cost)
}

fn optmise(w: &mut DVector<f32>, b: &mut f32, x_mx: &DMatrix<f32>,
           y_mx: &DVector<f32>, num_iterations: usize,
           learning_rate: f32)
        -> f32 {
    let mut cost = 0f32;
    for _ in 0..num_iterations {
        // Forward prop
        let ((dw, db), c) = propagate(w, *b, x_mx, y_mx);
        cost = c;

        // Update weights
        w.sub_assign(dw * learning_rate);
        b.sub_assign(db * learning_rate);
    }

    cost
}

fn predict(w: &DVector<f32>, b: f32, x_mx: &DMatrix<f32>)
        -> DVector<f32> {
    // Compute activation value
    let mut a_mx = (w.transpose() * x_mx).add_scalar(b).transpose();
    sigmoid(&mut a_mx.iter_mut());

    // Create vector Y of predictions
    let y_mx = DVector::from_iterator(a_mx.shape().0, a_mx
                .into_iter()
                .map(|&x|
                    if x < 0.5f32 {
                        0f32
                    }
                    else {
                        1f32
                    }));

    y_mx
}

fn main() {
    let data_file_name = "cifar-10-batches-bin/data_batch_1.bin";

    // Load CIFAR data
    let start_t = Instant::now();
    let (mut x_mx, y_mx) = load_cifar(data_file_name)
        .expect("Error loading the data!");
    let end_t = Instant::now();
    let elapsed = end_t.duration_since(start_t);
    println!("Time elapsed loading data : {:?}", elapsed);

    // Standardise data
    let start_t = Instant::now();
    x_mx = standardise_rgb(&x_mx, 255f32);
    let end_t = Instant::now();
    let elapsed = end_t.duration_since(start_t);
    println!("Time elapsed standardising data : {:?}", elapsed);


    // Setup weights and bias
    let mut w = DVector::<f32>::zeros(x_mx.shape().0);
    let mut b = 0f32;

    // Optimise weights
    let start_t = Instant::now();
    // Optimise w and b in place
    let final_cost = optmise(&mut w, &mut b, &x_mx, &y_mx, 2000, 0.005f32);
    let end_t = Instant::now();
    let elapsed = end_t.duration_since(start_t);
    println!("Time elapsed optimising data : {:?}", elapsed);

    // Predict
    let start_t = Instant::now();
    let y_hat = predict(&w, b, &x_mx);
    let end_t = Instant::now();
    let elapsed = end_t.duration_since(start_t);
    println!("Time elapsed predicting: {:?}", elapsed);

    let accuracy = 100f32 - (y_hat
        .iter()
        .zip(y_mx.iter())
        .map(|(yh, y)| na::abs(&(yh - y)))
        .sum::<f32>() / y_mx.nrows() as f32)
        * 100f32;

    println!("Final cost: {}", final_cost);
    println!("Accuracy on training set: {}", accuracy);
}
