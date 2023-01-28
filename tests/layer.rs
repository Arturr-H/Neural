#[cfg(test)]
mod tests {
    /* Imports */
    use neural_network::layer::Layer;

    /* Tests */
    #[test]
    fn initialize() -> () {
        let layer = Layer::new(5, 10);

        assert_eq!(layer.num_nodes_out() == 10, true);
        assert_eq!(layer.num_nodes_in() == 5, true);
        assert_eq!(layer.biases().len() == 10, true);
        assert_eq!(layer.weights().len() == 5, true);
    }
}