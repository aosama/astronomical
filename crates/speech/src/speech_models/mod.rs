//! Text-to-speech models, one folder per model family, the pieces both
//! families share beside them.
//!
//! A family folder owns its forward pass and its artifact layout. What every
//! family needs (reading checkpoint tensors as float32, the MLX layers with
//! no family-specific behaviour, and the error shape) lives here so the two
//! families cannot drift apart.

pub mod checkpoint_tensors;
pub mod engine_settings;
pub mod kokoro;
pub mod mlx_layers;
pub mod phonon;
pub mod speech_memory;
pub mod speech_model_error;
pub mod speech_request_slot;
#[cfg(test)]
pub mod speech_test_support;
pub mod voice_files;
