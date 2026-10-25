//! Direct-MLX contracts: the Metal encoder is held to the pure reference
//! encoder (which the integration lane holds to the Python tool), and the
//! requantizer is exercised end to end on a synthetic two-shard source small
//! enough to run in seconds. No real model artifacts.

mod encoder_contracts;
mod requantize_contracts;
mod support;
