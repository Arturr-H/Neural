/* Imports */
use crate::layer::GlobalNNFloatType;

/* Main */
pub struct Datapoint {
    /// Aka input layer
    inputs: Vec<GlobalNNFloatType>,

    /// Aka output layer
    expected: Vec<GlobalNNFloatType>,
}

/* Method implementations */
impl Datapoint {
    /* Getters */
    pub fn inputs(&self) -> &Vec<GlobalNNFloatType> { &self.inputs }
    pub fn expected(&self) -> &Vec<GlobalNNFloatType> { &self.expected }
}
