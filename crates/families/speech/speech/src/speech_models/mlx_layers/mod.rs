//! MLX layers both speech families build on. Convolution types stay with
//! their family: Kokoro's convolutions pad symmetrically and Phonon's are
//! causal, so only the weight-layout conversion they share lives here.

pub mod activations;
pub mod convolution_weight_layout;
pub mod layer_norm;
pub mod linear;
pub mod scalar_arithmetic;
