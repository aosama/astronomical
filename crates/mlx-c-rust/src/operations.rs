//! MLX operation wrappers, grouped by operation family.

pub(crate) mod activation;
pub(crate) mod array_creation;
pub(crate) mod array_shape;
pub(crate) mod attention;
pub(crate) mod convolution;
pub(crate) mod convolution_general;
pub(crate) mod creation;
pub(crate) mod cumulative;
pub(crate) mod elementwise_math;
pub(crate) mod fft;
pub(crate) mod general;
pub(crate) mod indexing;
pub(crate) mod linear_algebra;
pub(crate) mod math_binary;
pub(crate) mod math_unary;
pub(crate) mod normalization;
pub(crate) mod nvfp4;
pub(crate) mod padding;
pub(crate) mod quantization_construction;
pub(crate) mod quantized;
pub(crate) mod quantized_array;
pub(crate) mod random;
pub(crate) mod random_sampling;
pub(crate) mod reduction;
pub(crate) mod rope;
pub(crate) mod shape;
