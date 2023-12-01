use nalgebra::{SVector, DVector};

pub enum Activation {
    ReLU,
    Sigmoid,
    //Softmax
}

impl Activation {
    pub fn forward<const R: usize>(&self, input: &SVector<f32, R>) -> SVector<f32, R> {
        match self {
            Activation::ReLU => input.map(|x| if x > 0.0 { x } else { 0.0 }),
            Activation::Sigmoid => input.map(|x| 1.0 / (1.0 + (-x).exp())),
            /* Activation::Softmax => {
                let mut sum = 0.0;
                let mut out = SVector::<f32, R>::zeros();
                let max = input.max();

                for (i, x) in input.iter().enumerate() {
                    let exps = (x - max).exp();
                    out[i] = exps;
                    sum += exps;
                }

                out.map(|x| x / sum)
            } */
        }
    }

    pub fn dyn_forward(&self, input: &DVector<f32>) -> DVector<f32> {
        match self {
            Activation::ReLU => input.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect::<Vec<f32>>().into(),
            Activation::Sigmoid => input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<f32>>().into(),
            /* Activation::Softmax => {
                let mut sum = 0.0;
                let mut out = SVector::<f32, R>::zeros();
                let max = input.max();

                for (i, x) in input.iter().enumerate() {
                    let exps = (x - max).exp();
                    out[i] = exps;
                    sum += exps;
                }

                out.map(|x| x / sum)
            } */
        }
    }

    pub fn backward<const R: usize>(&self, input: &SVector<f32, R>) -> SVector<f32, R> {
        match self {
            Activation::ReLU => input.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let sig = self.forward(input);
                sig.map(|x| x * (1.0 - x))
            }
            //Activation::Softmax => 
        }
    }

    pub fn dyn_backward(&self, input: &DVector<f32>) -> DVector<f32> {
        match self {
            Activation::ReLU => input.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect::<Vec<f32>>().into(),
            Activation::Sigmoid => {
                let sig = self.dyn_forward(input);
                sig.iter().map(|x| x * (1.0 - x)).collect::<Vec<f32>>().into()
            }
            //Activation::Softmax => 
        }
    }
}