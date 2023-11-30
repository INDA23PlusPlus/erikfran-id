mod layers;

use layers::*;
use mnist::*;
use nalgebra::{self, DMatrix};

const HIDDEN_LAYERS: u8 = 2;
const LAYER_SIZE: [usize; HIDDEN_LAYERS as usize] = [128, 64];
const INPUTS: usize = 28 * 28;
const OUTPUTS: usize = 10;

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: u32 = 1000;

fn main() {
    // Deconstruct the returned Mnist struct.
   let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = DMatrix(trn_img.chunks(28 * 28));
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    let mut network: Vec<Box<dyn Layer>> = Vec::new();
    network.push(Box::new(Dense::<INPUTS, { LAYER_SIZE[0] }>::new()));
    network.push(Box::new(Convolutional::<{ LAYER_SIZE[0] }, { LAYER_SIZE[1] }>::new()));
    network.push(Box::new(Dense::<{ LAYER_SIZE[1] }, OUTPUTS>::new()));
}