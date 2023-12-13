#![feature(array_chunks)]
#![feature(iterator_try_collect)]
use layers::*;
use mnist::*;
use nalgebra::{self, DVector, SVector};
use activation::Activation;
use rand::prelude::*;
use std::{thread, env};

mod layers;
mod activation;

const RESOLUTION: usize = 28 * 28;

const HIDDEN_LAYERS: u8 = 3;
const LAYER_SIZE: [usize; HIDDEN_LAYERS as usize] = [500, 300, 75];
const INPUTS: usize = 28 * 28;
const OUTPUTS: usize = 10;

const EPOCHS: u32 = 1000;

pub const TEST_DATA_LENGTH: u32 = 10_000;
pub const TRAIN_DATA_LENGTH: u32 = 60_000;

const ACTIVATION: Activation = Activation::ReLU;

const MAX_SGD_BATCH_SIZE: usize = 1000;

const STACK_SIZE: usize = 32 * 1024 * 1024;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // Spawn thread with explicit stack size
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}

fn run() {
    // Deconstruct the returned Mnist struct.
   let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAIN_DATA_LENGTH)
        .test_set_length(TEST_DATA_LENGTH)
        //.download_and_extract()
        //.base_path("data\\")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize();

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

    let test_data = tst_img
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
    network.push(Box::new(Dense::<{ LAYER_SIZE[0] }, INPUTS>::new(ACTIVATION)));
    println!("Created layer 1.");
    //println!("weights: {:?}", network[0].weights());
    //println!("biases: {:?}", network[0].biases());
    network.push(Box::new(Dense::<{ LAYER_SIZE[1] }, { LAYER_SIZE[0] }>::new(ACTIVATION)));
    println!("Created layer 2.");
    network.push(Box::new(Dense::<{ LAYER_SIZE[2] }, { LAYER_SIZE[1] }>::new(ACTIVATION)));
    //println!("weights: {:?}", network[1].weights());
    //println!("biases: {:?}", network[1].biases());
    network.push(Box::new(Dense::<OUTPUTS, { LAYER_SIZE[2] }>::new(ACTIVATION)));
    println!("Created layer 3. With {}, {}", OUTPUTS, LAYER_SIZE[1]);
    //println!("weights: {:?}", network[2].weights());
    //println!("biases: {:?}", network[2].biases());

    println!("Created network.");

    let mut rng = rand::thread_rng();
    let mut sgd_current: usize = 0;
    let mut sgd_nums: Vec<usize> = (0..TRAIN_DATA_LENGTH as usize).collect();

    for epoch in 0..EPOCHS {
        let mut correct = 0;
        let mut loss = 0f32;

        // sgd
        let sgd_rand: f32 = rng.gen();
        let sgd_max = if sgd_current + (sgd_rand * MAX_SGD_BATCH_SIZE as f32) as usize > TRAIN_DATA_LENGTH as usize {
            sgd_nums.shuffle(&mut rng);
            sgd_current = 0;
            (sgd_rand * MAX_SGD_BATCH_SIZE as f32) as usize
        } else {
            sgd_current + (sgd_rand * MAX_SGD_BATCH_SIZE as f32) as usize
        };

        for i in sgd_nums[sgd_current..sgd_max].iter() {
            let image = &train_data[*i];
            let mut input = svector_to_dvector(image);

            for layer in network.iter_mut() {
                // println!("input len: {:?} of forward ", input.len());
                // println!("input: {:?}", input);
                input = layer.train_forward(&input);
            }

            let mut max = 0f32;
            let mut max_index = 0usize;
            let mut gradient_vector = DVector::<f32>::zeros(OUTPUTS);

            for (i, x) in input.iter().enumerate() {
                let mut mse = 0f32;
                if trn_lbl[i] as usize == i {
                    mse += x - 1f32;
                } else {
                    mse = *x;
                }

                loss += mse.powi(2);
                gradient_vector[i] = mse / 2f32;

                if *x > max {
                    max = *x;
                    max_index = i;
                }
            }

            if max_index == trn_lbl[*i] as usize {
                correct += 1;
            }

            let mut input = gradient_vector 
                .component_mul(&ACTIVATION.dyn_backward(&network.last().unwrap().get_z()));

            for layer in network.iter_mut().rev() {
                // println!("input len: {:?}", input.len());
                input = layer.backward(&input);
            }
        }

        network.iter_mut().for_each(|x| x.update((sgd_max - sgd_current) as u32));

        println!("Epoch: {} / {} Accuracy: {} Loss: {}", 
            epoch,
            EPOCHS,
            correct as f32 / TRAIN_DATA_LENGTH as f32, 
            loss / (TRAIN_DATA_LENGTH * OUTPUTS as u32) as f32);
    }

    let (accuracy, loss) = test_accuracy(&network, &test_data, &tst_lbl);
    println!("Test data: Accuracy: {} Loss: {}", accuracy, loss);
}

fn test_accuracy(network: &Vec<Box<dyn Layer>>, test_data: &Vec<SVector<f32, RESOLUTION>>, test_labels: &Vec<u8>) -> (f32, f32) {
    let mut correct = 0;
    let mut loss = 0f32;

    for (i, image) in test_data.iter().enumerate() {
        let mut input = svector_to_dvector(image);

        for layer in network.iter() {
            input = layer.forward(&input);
        }

        let mut max = 0f32;
        let mut max_index = 0usize;

        for (i, x) in input.iter().enumerate() {
            let mut mse = 0f32;
            if test_labels[i] as usize == i {
                mse += (x - 1f32).powi(2);
            } else {
                mse = x.powi(2);
            }

            loss += mse;

            if *x > max {
                max = *x;
                max_index = i;
            }
        }

        if max_index == test_labels[i] as usize {
            correct += 1;
        }

    }

    (correct as f32 / TEST_DATA_LENGTH as f32, loss / (TEST_DATA_LENGTH * OUTPUTS as u32) as f32)
}