/* Imports */
use crate::activations::{ self, Activation };

/// `f32` (single precision) is faster and uses less memory
/// than f64 (double precision), but f64 has higher precision
/// and is more suitable for certain types of computations.
/// So might switch between these, that's why I made a type
pub type GlobalNNFloatType = f64;

/* Main */
pub struct Layer {
    // TODO: Change usize to smaller like u8 / u16 for performanmce
    /// `num_nodes_in` is the amount of nodes in the
    /// __*previous*__ layer
    num_nodes_in: usize,
    
    /// `num_nodes_out` is the amount of nodes in the
    /// __*current*__ layer - Aka nodes pointing outwards
    num_nodes_out: usize,

    /// The weights are saved in a 2d matrix; 1st __"dimension"__
    /// is every nodes in the previous layer, and the 2nd
    /// __"dimension"__  is every weight from that node to each and
    /// every node in the current layer
    weights: Vec<Vec<GlobalNNFloatType>>,

    /// The biases are a one-dimensional vector, because the biases
    /// represent each nodes' biases in the current layer.
    biases: Vec<GlobalNNFloatType>,

    /// The activation function represented by this layer
    activation: Activation
}

/* Method implementations */
impl Layer {
    /// Layer constructor
    pub fn new(num_nodes_in: usize, num_nodes_out: usize) -> Self {
        Self {
            num_nodes_in,
            num_nodes_out,
            weights: vec![vec![0.;num_nodes_out];num_nodes_in],
            biases: vec![0.;num_nodes_out],
            activation: activations::step
        }
    }

    /// Calculate outputs of layer
    pub fn calculate_outputs(&self, inputs: Vec<GlobalNNFloatType>) -> Vec<GlobalNNFloatType> {
        let mut activations:Vec<GlobalNNFloatType> = Vec::with_capacity(self.num_nodes_out);

        /* Iterate over the *Current* layer */
        for node_out in 0..self.num_nodes_out {

            /* Grab the bias which the current node possesses */
            let mut weighted_input = self.biases[node_out];

            /* Iterate over ingoing nodes */
            for node_in in 0..self.num_nodes_in {

                /* 
                    Here we caculate the output value of each node.
                    The `weighted_input` variable is increased by 
                    all nodes in the previous layers multiplied by
                    their respective weights (aka influence on current)
                */
                weighted_input += inputs[node_in] * self.weights[node_in][node_out];
            };

            /* Set the value inside of the `activations` vec we created */
            activations[node_out] = self.activation()(weighted_input);
        }

        /* Return */
        activations
    }

    /* GETTERS */
    pub fn num_nodes_in(&self) -> usize { self.num_nodes_in }
    pub fn num_nodes_out(&self) -> usize { self.num_nodes_out }
    pub fn weights(&self) -> &Vec<Vec<f64>> { &self.weights }
    pub fn biases(&self) -> &Vec<f64> { &self.biases }
    pub fn activation(&self) -> Activation { self.activation }
}
