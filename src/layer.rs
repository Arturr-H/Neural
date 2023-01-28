/* Imports */

/// `f32` (single precision) is faster and uses less memory
/// than f64 (double precision), but f64 has higher precision
/// and is more suitable for certain types of computations.
/// So might switch between these, that's why I made a type
pub type GlobalNNFloatType = f64;

/* Main */
struct Layer {
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
    biases: Vec<GlobalNNFloatType>
}
