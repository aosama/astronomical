//! Text-to-speech models, one folder per model family, the pieces both
//! families share beside them.
//!
//! A family folder owns its forward pass and its artifact layout. What every
//! family needs (reading checkpoint tensors as float32, the MLX layers with
//! no family-specific behaviour, and the error shape) lives here so the two
//! families cannot drift apart.

pub(crate) mod checkpoint_tensors;
pub(crate) mod engine_settings;
pub(crate) mod kokoro;
pub(crate) mod mlx_layers;
pub(crate) mod phonon;
pub(crate) mod speech_memory;
pub(crate) mod speech_model_error;
pub(crate) mod speech_request_slot;
#[cfg(test)]
pub(crate) mod speech_test_support;
pub(crate) mod voice_files;
