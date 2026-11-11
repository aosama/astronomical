//! MLX layers both speech families build on. Convolution types stay with
//! their family: Kokoro's convolutions pad symmetrically and Phonon's are
//! causal, so only the weight-layout conversion they share lives here.

pub(crate) mod activations;
pub(crate) mod convolution_weight_layout;
pub(crate) mod layer_norm;
pub(crate) mod linear;
pub(crate) mod scalar_arithmetic;
