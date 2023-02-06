use crate::layer::{GlobalNNFloatType, Layer};

/* Structs */
pub struct NetworkLearnData {
    pub layer_data: Vec<LayerLearnData>
}
pub struct LayerLearnData {
	pub inputs: Vec<GlobalNNFloatType>,
	pub weighted_inputs: Vec<GlobalNNFloatType>,
	pub activations: Vec<GlobalNNFloatType>,
	pub node_values: Vec<GlobalNNFloatType>,
}

/* Method implementations */
impl NetworkLearnData {
    pub fn new(layers: Vec<Layer>) -> Self {
        let mut layer_data:Vec<LayerLearnData> = Vec::new();
		for i in 0..layers.len() {
            layer_data.push(LayerLearnData::new(&layers[i]));
		}

        Self { layer_data }
	}
}   
impl LayerLearnData {
    pub fn new(layer: &Layer) -> Self {
        Self {
            weighted_inputs: vec![0.; layer.num_nodes_out()],
            activations: vec![0.; layer.num_nodes_out()],
            node_values: vec![0.; layer.num_nodes_out()],
            inputs: vec![0.; layer.num_nodes_in()]
        }
	}
}
