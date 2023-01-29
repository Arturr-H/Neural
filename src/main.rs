/* Global allowings */
#![allow(
    dead_code,
    unused_imports,
    unused_variables
)]

/* Modules */
pub mod layer;
pub mod network;
pub mod activations;

/* Imports */

/* Main */
fn main() {
    network::Network::new(&[1, 2, 3, 4]);
}
