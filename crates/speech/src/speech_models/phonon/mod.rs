//! Phonon 31M text-to-speech on MLX, ported from Gradium's reference runtime
//! (gradium-ai/xn-ptts, MIT or Apache-2.0), itself built on Kyutai's Pocket
//! TTS: text and a voice in, 24 kHz audio out.
//!
//! A causal flow language model turns the voice prompt and the text tokens
//! into one Mimi latent per 80 ms frame, and the Mimi codec's decoder turns
//! those latents into samples.

mod artifact_files;
mod attention_history;
mod causal_attention_mask;
mod causal_convolutions;
mod causal_transformer;
mod configuration;
mod engine;
mod flow_head;
mod flow_language_model;
mod frame_conditioning;
mod latent_generation;
mod mimi_decoder;
mod rotary_embedding;
mod sinusoidal_embedding;
mod speech_model;
mod text_chunks;
mod text_tokenizer;
mod time_embedding;
mod voice;

pub use engine::PhononSpeechEngine;
