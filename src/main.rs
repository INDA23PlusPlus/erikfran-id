const HIDDEN_LAYERS: u8 = 2;
const LAYER_SIZE: [u32; HIDDEN_LAYERS as usize] = [128, 64];
const INPUTS: u32 = 128 * 128;
const OUTPUTS: u32 = 10;

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: u32 = 1000;

mod network;

fn main() {
    println!("Hello, world!");
}
