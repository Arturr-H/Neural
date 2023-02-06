/* Imports */
use crate::{activations::{ self, Activation }, learndata::LayerLearnData};
use rand::{ self, Rng };
use std::fmt::Display;

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
    /// every node in the current layer. It's actually stored in 1d
    /// for performance reasons
    weights: Vec<GlobalNNFloatType>,

    /// The biases are a one-dimensional vector, because the biases
    /// represent each nodes' biases in the current layer.
    biases: Vec<GlobalNNFloatType>,

    /// The activation function represented by this layer
    activation: Activation,

    /// The cost gradient weights are a 2d matrix, responding to
    /// each and every weight in the `weights` matrix. The value
    /// is determined by how much the cost changes in response to
    /// a change in the weight. Stored in 1d vector
    cost_gradient_weights: Vec<GlobalNNFloatType>,

    /// The cost gradient biases is an array responding to
    /// each and every bias in the `biases` array. The value
    /// is determined by how much the cost changes in response to
    /// a change in the bias.
    cost_gradient_biases: Vec<GlobalNNFloatType>,
}

/* Method implementations */
impl Layer {
    /// Layer constructor
    pub fn new(num_nodes_in: usize, num_nodes_out: usize) -> Self {
        Self {
            num_nodes_in,
            num_nodes_out,

            /* Weights */
            weights:               vec![0.; num_nodes_out * num_nodes_in],
            cost_gradient_weights: vec![0.; num_nodes_out * num_nodes_in],

            /* Biases */
            biases:               vec![0.;num_nodes_out],
            cost_gradient_biases: vec![0.;num_nodes_out],

            // TODO: Change this activation
            activation: activations::sigmoid
        }
        
        /* Initialize fields */
        .initialize_weights()
        .initialize_biases()
    }

    /// Calculate outputs of layer
    pub fn calculate_outputs(&self, inputs: &Vec<GlobalNNFloatType>) -> Vec<GlobalNNFloatType> {
        let mut weighted_inputs:Vec<GlobalNNFloatType> = Vec::with_capacity(self.num_nodes_out());

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
                weighted_input += inputs[node_in] * self.get_weight(node_in, node_out);
            };

            weighted_inputs.push(weighted_input);

        }

        /* Apply activation to all values */
        let mut activations = Vec::with_capacity(self.num_nodes_out());
        for output_node in 0..self.num_nodes_out() {
            activations.push(self.activation()(&weighted_inputs, output_node));
        }

        /* Return */
        activations
    }

    /// Get a value from the `weights` "matrix"
	pub fn get_weight(&self, node_in: usize, node_out: usize) -> f64 {
		let flat_index = node_out * self.num_nodes_in() + node_in;
		self.weights[flat_index]
	}

    /// Get a mutable reference to a value in the `weights` "matrix"
	pub fn get_weight_mut(&mut self, node_in: usize, node_out: usize) -> &mut f64 {
		let flat_index = node_out * self.num_nodes_in() + node_in;
		&mut self.weights[flat_index]
	}

    /// Get a value from the `cost_gradient_weights` "matrix"
	pub fn get_cost_gradient_weight(&self, node_in: usize, node_out: usize) -> f64 {
		let flat_index = node_out * self.num_nodes_in() + node_in;
		self.cost_gradient_weights[flat_index]
	}

    /// Get a mutable reference to a value in the `cost_gradient_weights` "matrix"
	pub fn get_cost_gradient_weight_mut(&mut self, node_in: usize, node_out: usize) -> &mut f64 {
		let flat_index = node_out * self.num_nodes_in() + node_in;
		&mut self.cost_gradient_weights[flat_index]
	}

    /// Node cost of a single neuron. Compares it to expected
    pub fn node_cost(neuron: GlobalNNFloatType, expected: GlobalNNFloatType) -> GlobalNNFloatType {
        let err = neuron - expected;
        err * err
    }

    /// Initialize weights
    fn initialize_weights(mut self) -> Self {
        let mut rng = rand::thread_rng();
        
        for node_in in 0..self.num_nodes_in {
            for node_out in 0..self.num_nodes_out {
                let v = rng.gen_range((-1.0)..(1.0));
                *self.get_weight_mut(node_in, node_out) = v / (self.num_nodes_in as GlobalNNFloatType).sqrt();
            };
        };

        self
    }
    /// Initialize biases
    fn initialize_biases(mut self) -> Self {
        let mut rng = rand::thread_rng();

        for node_out in 0..self.num_nodes_out {
            let v = rng.gen_range((-1.0)..(1.0));
            self.biases[node_out] = v;
        };

        self
    }

    /// Update all weights and biases depending on the cost gradients.
    /// (Gradient descent)
    pub fn apply_gradients(&mut self, learn_rate: f64) -> () {
        for node_out in 0..self.num_nodes_out {
            self.biases[node_out] -= self.cost_gradient_biases[node_out];
            
            for node_in in 0..self.num_nodes_in {
                *self.get_weight_mut(node_in, node_out) -= self.get_cost_gradient_weight(node_in, node_out) * learn_rate;
            };
        };
    }

    /* [GETTERS] General */
    pub fn num_nodes_in(&self) -> usize { self.num_nodes_in }
    pub fn num_nodes_out(&self) -> usize { self.num_nodes_out }
    pub fn activation(&self) -> Activation { self.activation }
    
    /* [GETTERS] Weights and biases */
    pub fn weights(&self) -> &Vec<GlobalNNFloatType> { &self.weights }
    pub fn biases(&self) -> &Vec<GlobalNNFloatType> { &self.biases }
    pub fn weights_mut(&mut self) -> &mut Vec<GlobalNNFloatType> { &mut self.weights }
    pub fn biases_mut(&mut self) -> &mut Vec<GlobalNNFloatType> { &mut self.biases }

    /* [GETTERS] Cost gradient */
    pub fn cost_gradient_biases(&self) -> &Vec<GlobalNNFloatType> { &self.cost_gradient_biases }
    pub fn cost_gradient_weights(&self) -> &Vec<GlobalNNFloatType> { &self.cost_gradient_weights }
    pub fn cost_gradient_biases_mut(&mut self) -> &mut Vec<GlobalNNFloatType> { &mut self.cost_gradient_biases }
    pub fn cost_gradient_weights_mut(&mut self) -> &mut Vec<GlobalNNFloatType> { &mut self.cost_gradient_weights }
}

/* Debug implementation */
impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /* Write to formatter */
        let mut string = format!("\n[Layer {}:{}]", self.num_nodes_in, self.num_nodes_out);

        /* Write every neuron and its weights like this: bias:[weight1, weight2, ...] */
        for node_out in 0..self.num_nodes_out {

            /* Round weights and biases to three decimals */
            string.push_str(&format!(
                "\n    Neuron {}: bias:{:.3}, ",
                node_out,
                self.biases[node_out]
            ));

            string.push_str(" weights:[");
            for node_in in 0..self.num_nodes_in {
                string.push_str(&format!("{:.3}, ", self.get_weight(node_in, node_out)));
            };

            string.push_str("]");
        };

        write!(f, "{}", string)
    }
}
