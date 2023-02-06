/* Imports */
use crate::layer::GlobalNNFloatType;
pub type Activation = fn(&Vec<GlobalNNFloatType>, usize) -> GlobalNNFloatType;

/// Step
pub fn step(input: GlobalNNFloatType) -> GlobalNNFloatType {
    if input > 0. { 1. } else { 0. }
}

/// Sigmoid
pub fn sigmoid(inputs: &Vec<GlobalNNFloatType>, index: usize) -> GlobalNNFloatType {
    return 1.0 / (1. + (-inputs[index]).exp());
}
pub fn sigmoid_derivative(inputs: &Vec<GlobalNNFloatType>, index: usize) -> GlobalNNFloatType {
    let a = sigmoid(inputs, index);
    return a * (1. - a);
}

/// SiLU
pub fn silu(input: GlobalNNFloatType) -> GlobalNNFloatType {
    input / (1. + (-input).exp())
}
