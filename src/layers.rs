use std::usize;

use nalgebra::{SMatrix, SVector, DVector};
use rand::distributions::Uniform;

pub trait Layer {
    fn forward(&self, input: DVector<f32>) -> DVector<f32>;
}

pub struct Dense<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, R>
}

impl <const R: usize, const C: usize> Dense<R, C> {
    pub fn new() -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        Dense {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, R>::from_distribution(&between, &mut rng),
        }
    }
}

impl<const R: usize, const C: usize> Layer for Dense<R, C> {
    fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        (self.weights * input + self.biases).data.as_slice().to_vec().into()
    }
}

pub struct Convolutional<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, R>
}

impl <const R: usize, const C: usize> Convolutional<R, C> {
    pub fn new() -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        Convolutional {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, R>::from_distribution(&between, &mut rng),
        }
    }
}

impl<const R: usize, const C: usize> Layer for Convolutional<R, C> {
    fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        (self.weights * input + self.biases).data.as_slice().to_vec().into()
    }
}