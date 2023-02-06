/* Global allowings */
#![allow(
    dead_code,
    unused_imports,
    unused_variables
)]

use crate::datapoint::Datapoint;

/* Modules */
pub mod layer;
pub mod network;
pub mod learndata;
pub mod datapoint;
pub mod activations;

/* Imports */

/* Main */
fn main() {
    let mut nn = network::Network::new(&[784, 6, 8, 6, 2]);
    let training_data = vec![
        datapoint::Datapoint { inputs: vec![0., 0., 0., 0.], expected: vec![0., 0.] },

        datapoint::Datapoint { inputs: vec![1., 0., 0., 0.], expected: vec![1., 0.] },
        datapoint::Datapoint { inputs: vec![1., 1., 0., 0.], expected: vec![1., 0.] },
        datapoint::Datapoint { inputs: vec![0., 1., 0., 0.], expected: vec![1., 0.] },

        datapoint::Datapoint { inputs: vec![0., 0., 1., 0.], expected: vec![0., 1.] },
        datapoint::Datapoint { inputs: vec![0., 0., 1., 1.], expected: vec![0., 1.] },
        datapoint::Datapoint { inputs: vec![0., 0., 0., 1.], expected: vec![0., 1.] },

        datapoint::Datapoint { inputs: vec![0., 1., 1., 1.], expected: vec![1., 1.] },
        datapoint::Datapoint { inputs: vec![1., 1., 1., 0.], expected: vec![1., 1.] },
        datapoint::Datapoint { inputs: vec![1., 1., 0., 1.], expected: vec![1., 1.] }
    ];
    println!("{}", nn);

    let training_data = Datapoint::load_mnist();
    for i in 0..1000 {
        nn.learn(&training_data, 0.0001);
        dbg!(nn.cost());
    }
}
