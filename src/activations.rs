/* Activation functions */

/// Step
pub fn step(input: GlobalNNFloatType) -> GlobalNNFloatType {
    if input > 0 { 1 } else { 0 }
}
