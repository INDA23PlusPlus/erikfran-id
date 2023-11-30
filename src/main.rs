#![feature(array_chunks)]
#![feature(iterator_try_collect)]
mod layers;

use layers::*;
use mnist::*;
use nalgebra::{self, DMatrix, DVector, SVector};
use std::{env, fs::{self, File}, io::Read};

const RESOLUTION: usize = 28 * 28;

const HIDDEN_LAYERS: u8 = 2;
const LAYER_SIZE: [usize; HIDDEN_LAYERS as usize] = [28, 10];
const INPUTS: usize = 28 * 28;
const OUTPUTS: usize = 10;

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: u32 = 1;

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");

    // Deconstruct the returned Mnist struct.
   let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        //.download_and_extract()
        .base_path("data\\")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize();

/*     let path = "data\\train-images-idx3-ubyte";

    let mut content: Vec<u8> = Vec::new();
    let trn_img = {
        let mut fh = File::open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to images at {}.", path));
        let _ = fh
            .read_to_end(&mut content)
            .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({:?})", path));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

    //let trn_img = fs::read("data\\train-images-idx3-ubyte").unwrap();
    let trn_lbl = fs::read("data\\train-labels-idx1-ubyte").unwrap();
    let tst_img = fs::read("data\\t10k-images-idx3-ubyte").unwrap();
    let tst_lbl = fs::read("data\\t10k-labels-idx1-ubyte").unwrap(); */

    println!("Loaded MNIST data.");

    let train_data = trn_img
        .array_chunks::<RESOLUTION>()
        .map(|x|{
            let mut array = [0f32; RESOLUTION];

            x.iter()
                .enumerate()
                .for_each(|(i, x)| array[i] = *x as f32 / 256f32);

            SVector::from(array)
        })
        .collect::<Vec<SVector<f32, RESOLUTION>>>();

    println!("Loaded training data.");

    let _test_data = tst_img
        .array_chunks::<RESOLUTION>()
        .map(|x|{
            let mut array = [0f32; RESOLUTION];

            x.iter()
                .enumerate()
                .for_each(|(i, x)| array[i] = *x as f32 / 256f32);

            SVector::from(array)
        })
        .collect::<Vec<SVector<f32, RESOLUTION>>>();

    println!("Loaded test data.");

    let mut network: Vec<Box<dyn Layer>> = Vec::new();
    println!("Creating network.");
    network.push(Box::new(Dense::<INPUTS, { LAYER_SIZE[0] }>::new()));
    println!("Created layer 1.");
    network.push(Box::new(Dense::<{ LAYER_SIZE[0] }, { LAYER_SIZE[1] }>::new()));
    println!("Created layer 2.");
    network.push(Box::new(Dense::<{ LAYER_SIZE[1] }, OUTPUTS>::new()));

    println!("Created network.");

    for epoch in 0..EPOCHS {
        let mut correct = 0;
        let mut total = 0;
        println!("Epoch: {}", epoch);

        for (i, image) in train_data.iter().enumerate() {
            let mut output = dvector_from_svector(image);

            for layer in network.iter() {
                output = layer.forward(&output);
            }

            let mut max = 0f32;
            let mut max_index = 0usize;

            for (i, x) in output.iter().enumerate() {
                if *x > max {
                    max = *x;
                    max_index = i;
                }
            }

            if max_index == trn_lbl[i] as usize {
                correct += 1;
            }

            total += 1;
        }

        println!("Epoch: {} Accuracy: {}", epoch, correct as f32 / total as f32);
    }
}