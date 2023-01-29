/* Imports */
use crate::layer::GlobalNNFloatType;
pub type Activation = fn(GlobalNNFloatType) -> GlobalNNFloatType;

/// Step
pub fn step(input: GlobalNNFloatType) -> GlobalNNFloatType {
    if input > 0. { 1. } else { 0. }
}

/// Sigmoid
pub fn sigmoid(input: GlobalNNFloatType) -> GlobalNNFloatType {
    1. / (1. + (-input).exp())
}

/// SiLU
pub fn silu(input: GlobalNNFloatType) -> GlobalNNFloatType {
    input / (1. + (-input).exp())
}
