use nalgebra::{SMatrix, SVector, DVector};
use rand::distributions::Uniform;

pub trait Layer {
    fn forward(&self, input: DVector<f32>) -> DVector<f32>;
}

pub struct Dense<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, C>
}

impl <const R: usize, const C: usize> Dense<R, C> {
    pub fn new(&self) -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        Dense {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, C>::from_distribution(&between, &mut rng),
        }
    }
}

impl<const R: usize, const C: usize> Layer for Dense<R, C> {
    fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        todo!()//self.weights * input + self.biases
    }
}

pub struct Convolutional<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, C>
}

impl <const R: usize, const C: usize> Convolutional<R, C> {
    pub fn new(&self) -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        Convolutional {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, C>::from_distribution(&between, &mut rng),
        }
    }
}

impl<const R: usize, const C: usize> Layer for Convolutional<R, C> {
    fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        todo!()//self.weights * input + self.biases
    }
}