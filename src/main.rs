mod layers;

use layers::*;

const HIDDEN_LAYERS: u8 = 2;
const LAYER_SIZE: [usize; HIDDEN_LAYERS as usize] = [128, 64];
const INPUTS: usize = 128 * 128;
const OUTPUTS: usize = 10;

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: u32 = 1000;

fn main() {
    let mut network: Vec<Box<dyn Layer>> = Vec::new();
    network.push(Box::new(Dense::<INPUTS, { LAYER_SIZE[0] }>::new()));
    network.push(Box::new(Convolutional::<{ LAYER_SIZE[0] }, { LAYER_SIZE[1] }>::new()));
    network.push(Box::new(Dense::<{ LAYER_SIZE[1] }, OUTPUTS>::new()));
}