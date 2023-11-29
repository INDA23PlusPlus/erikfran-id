use nalgebra::{SMatrix, SVector};

pub struct Layer<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, C>
}

const fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const fn layers<const N: usize, const S: [u32; N]>(sizes: [u32; N]) -> [Layer<S[i], S[i+1]>; N] {
    let mut layers = [Layer::new(); N];
    for i in 0..N {
        layers[i] = Layer::new(SMatrix::new_random(S[i], S[i+1]), SVector::new_random(S[i+1]));
    }
    layers
}

pub struct Network<const N: usize, const S: [u32; N]> {
    layers: Vec<Layer<>>,
}

impl Network {
    pub fn new() -> Network {
        Network { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.feed_forward(&output);
        }
        output
    }
}

impl Layer {
    pub fn new(weights: Vec<Vec<f64>>, biases: Vec<Vec<f64>>) -> Layer {
        Layer {
            weights: weights,
            biases: biases,
        }
    }

    pub fn feed_forward(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = Vec::new();
        for neuron in &self.weights {
            let mut neuron_output = 0.0;
            for (i, weight) in neuron.iter().enumerate() {
                neuron_output += weight * input[i];
            }
            output.push(neuron_output);
        }
        output
    }
}