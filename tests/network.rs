#[cfg(test)]
mod tests {
    /* Imports */
    use neural_network::network::Network;

    /* Tests */
    #[test]
    fn construct() -> () {
        let nn = Network::new(&[2, 4, 5, 2]);
        assert_eq!(nn.layers()[0].num_nodes_out() == 2, true);
        assert_eq!(nn.layers()[1].num_nodes_out() == 4, true);
        assert_eq!(nn.layers().len() == 4, true);
    }

    #[test]
    fn calculate_outputs() -> () {
        let nn = Network::new(&[2, 4, 5, 2]);
        nn.calculate_outputs(vec![1., 0.]);
    }

    #[test]
    fn classify() -> () {
        let nn = Network::new(&[2, 4, 5, 2]);
        nn.classify(vec![1., 0.]);
    }
}
