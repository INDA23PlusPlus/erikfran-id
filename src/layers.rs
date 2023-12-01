use std::usize;
use nalgebra::{SMatrix, SVector, DVector, DMatrix, VecStorage, Dyn, Matrix, MatrixView};
use rand::distributions::Uniform;
use crate::activation::Activation;

const LEARNING_RATE: f32 = 0.01;

pub fn svector_to_dvector<const R: usize>(vector: &SVector<f32, R>) -> DVector<f32> {
    vector.data
        .as_slice()
        .to_vec()
        .into()
}

pub fn dvector_to_svector<const R: usize>(vector: &DVector<f32>) -> SVector<f32, R> {
    SVector::<f32, R>::from_iterator(vector.iter().cloned())
}

pub trait Layer {
    fn forward(&self, input: &DVector<f32>) -> DVector<f32>;
    fn train_forward(&mut self, input: &DVector<f32>) -> DVector<f32>;
    fn backward(&mut self, input: &DVector<f32>) -> DVector<f32>;
    fn update(&mut self, data_len: u32);
    fn get_z(&self) -> DVector<f32>;
}

pub struct Dense<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, R>,
    weights_gradient: SMatrix<f32, R, C>,
    biases_gradient: SVector<f32, R>,
    cached_input: DVector<f32>,
    cached_z: SVector<f32, R>,
    activation: Activation,
}

impl <const R: usize, const C: usize> Dense<R, C> {
    pub fn new(activation: Activation) -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        Dense {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, R>::from_distribution(&between, &mut rng),
            weights_gradient: SMatrix::<f32, R, C>::zeros(),
            biases_gradient: SVector::<f32, R>::zeros(),
            cached_input: DVector::<f32>::zeros(R),
            cached_z: SVector::<f32, R>::zeros(),
            activation
        }
    }
}

impl<const R: usize, const C: usize> Layer for Dense<R, C> {
    fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        svector_to_dvector(
            &self.activation.forward(
                &(self.weights * input + self.biases)))
    }

    fn train_forward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.cached_input = input.clone();
        let z = &(self.weights * input + self.biases);
        self.cached_z = *z;

        svector_to_dvector(&self.activation.forward(z))
    }

    fn backward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        /* let delta = input * self.activation.backward(
            &dvector_to_svector(&self.cached_input)); */

        println!("{} == {}", self.biases_gradient.len(), input.len());

        self.biases_gradient += input;

        println!("{}, {} == {}, {} == {}, {}", 
            self.weights_gradient.nrows(), self.weights_gradient.ncols(), 
            input.len(), input.ncols(),
            self.cached_input.len(), self.cached_z.ncols());

        self.weights_gradient += input * self.cached_input.clone().transpose();

        svector_to_dvector(&(self.weights.transpose() * input)
                .component_mul(
                    &svector_to_dvector(&self.activation.backward(&self.cached_z))
                ))
    }

    fn get_z(&self) -> DVector<f32> {
        svector_to_dvector(&self.cached_z)
    }

    fn update(&mut self, data_len: u32) {
        self.weights -= (LEARNING_RATE * self.weights_gradient) / data_len as f32;
        self.biases -=  (LEARNING_RATE * self.biases_gradient) / data_len as f32;
    }
}

/* pub struct LinearDense<const R: usize, const C: usize> {
    weights: SMatrix<f32, R, C>,
    biases: SVector<f32, R>,
    weights_gradient: SMatrix<f32, R, C>,
    biases_gradient: SVector<f32, R>,
    cached_input: DVector<f32>,
    cached_output: DVector<f32>,
}

impl <const R: usize, const C: usize> LinearDense<R, C> {
    pub fn new() -> Self {
        let between = Uniform::from(0.0f32..1.0f32);
        let mut rng = rand::thread_rng();

        LinearDense {
            weights: SMatrix::<f32, R, C>::from_distribution(&between, &mut rng),
            biases: SVector::<f32, R>::from_distribution(&between, &mut rng),
            weights_gradient: SMatrix::<f32, R, C>::zeros(),
            biases_gradient: SVector::<f32, R>::zeros(),
            cached_input: DVector::<f32>::zeros(R),
            cached_output: DVector::<f32>::zeros(R),
        }
    }
}

impl<const R: usize, const C: usize> Layer for LinearDense<R, C> {
    fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        svector_to_dvector(&(self.weights * input + self.biases))
    }

    fn train_forward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.cached_input = input.clone();
        self.cached_output = svector_to_dvector(&(self.weights * input + self.biases));

        self.cached_output.clone()
    }

    fn backward(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.biases_gradient += input;

        //self.weights_gradient += ;

        svector_to_dvector(&(self.weights.transpose() * input))
    }

    fn update(&mut self, data_len: u32) {
        self.weights -= (LEARNING_RATE * self.weights_gradient) / data_len as f32;
        self.biases -=  (LEARNING_RATE * self.biases_gradient) / data_len as f32;
    }
} */