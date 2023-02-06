/* Imports */
use crate::layer::GlobalNNFloatType;
use rust_mnist::Mnist;

/* Main */
pub struct Datapoint {
    /// Aka input layer
    pub inputs: Vec<GlobalNNFloatType>,

    /// Aka output layer
    pub expected: Vec<GlobalNNFloatType>,
}

/* Method implementations */
impl Datapoint {
    /// Loads the MNIST dataset
    pub fn load_mnist<'a>() -> Vec<Datapoint> {
        let mnist = Mnist::new("./data/");
        let mut labels = Vec::new();

        for (label, data) in mnist.train_labels.iter().zip(mnist.train_data.iter()) {
            labels.push(Datapoint {
                inputs: data.to_vec().iter().map(|e| *e as f64).collect::<Vec<f64>>(),
                expected: to_byte_array_u8(*label).to_vec().iter().map(|e| *e as f64).collect::<Vec<f64>>(),
            })
        };

        labels
    }

    /* Getters */
    pub fn inputs(&self) -> &Vec<GlobalNNFloatType> { &self.inputs }
    pub fn expected(&self) -> &Vec<GlobalNNFloatType> { &self.expected }
}

fn to_byte_array_u8(value: u8) -> [u8; 8] {
    let mut array = [0; 8];

    for i in 0..8 {
        array[7 - i] = (value >> i) & 1;
    }

    array
}
