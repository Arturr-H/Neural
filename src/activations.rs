pub type Activation = fn(GlobalNNFloatType) -> GlobalNNFloatType;

/// Step
pub fn step(input: GlobalNNFloatType) -> GlobalNNFloatType {
    if input > 0 { 1 } else { 0 }
}
