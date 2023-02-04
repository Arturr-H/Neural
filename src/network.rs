/* Imports */
use crate::{ layer::{ Layer, GlobalNNFloatType }, datapoint::Datapoint };
use std::fmt::Display;

/* Main */
pub struct Network {
    /// Each individual layer is stored here, including output and input
    layers: Vec<Layer>
}

/* Method implementations */
impl Network {
    /* Constructor */
    /// Construct a new neural network with specified layer sizes.
    /// 
    /// ## Example
    /// ```
    /// use neural_network::network::Network;
    /// let nn = Network::new(&[2, 6, 8, 6, 2]);
    /// ```
    pub fn new(layer_sizes: &[usize]) -> Self {
        let size = layer_sizes.len();
        let mut layers:Vec<Layer> = Vec::with_capacity(size);
        for index in 0..size {

            /* We can't grab the previous input nodes if we are at the first index */
            let into_prev;
            if index == 0 { into_prev = 0 }
            else { into_prev = layer_sizes[index - 1] }
            layers.push(Layer::new(into_prev, layer_sizes[index]));
        };

        Network { layers }
    }

    /// Calculate outputs - from the beginning we pass
    /// inputs through the first layer - then the
    /// outputs of that layer to next layer and so on
    pub fn calculate_outputs(&self, mut inputs: Vec<GlobalNNFloatType>) -> Vec<GlobalNNFloatType> {
        for layer in &self.layers {
            inputs = layer.calculate_outputs(inputs);
        }

        return inputs
    }

    /// Calculates outputs and classifies which output 
    /// neuron has highest value (index of)
    pub fn classify(&self, inputs: Vec<GlobalNNFloatType>) -> usize {
        let outputs = self.calculate_outputs(inputs);
        let (mut a, mut b):(usize, GlobalNNFloatType) = (0, 0.);
        for (index, output) in outputs.iter().enumerate() {
            if output > &b { a = index; b = *output; }
        };

        a
    }

    /// The total cost of a *single* `DataPoint`
    /// Calculates the output of the `Network` once provided with the
    /// `inputs` field from `DataPoint`. Iterates over each output neuron
    /// and compares it to every item in the `expected` field of `DataPoint`
    pub fn cost_single(&self, datapoint: Datapoint) -> GlobalNNFloatType {
        let recieved_outputs = self.calculate_outputs(datapoint.inputs);
        let mut cost:GlobalNNFloatType = 0.;

        for (index, recv) in recieved_outputs.iter().enumerate() {
            cost += Layer::node_cost(*recv, datapoint.expected[index]);
        }

        cost
    }

    /// The total cost of *multiple* `DataPoint`s
    pub fn cost_multiple(&self, datapoints: Vec<Datapoint>) -> GlobalNNFloatType {
        let mut total_cost:GlobalNNFloatType = 0.;
        let datapoints_len = datapoints.len();

        for datapoint in datapoints {
            total_cost += self.cost_single(datapoint);
        }

        total_cost / datapoints_len as GlobalNNFloatType
    }

    /* Getters */
    pub fn layers(&self) -> &Vec<Layer> { &self.layers }
}

/* Debug impl */
impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layers = self.layers().iter().map(|e| format!("{e}")).collect::<Vec<String>>().join("\n  ");
        write!(f,
            "\n#################### NETWORK START{}\n#################### NETWORK END",
            layers
        )
    }
}
