//! Kokoro-82M text-to-speech on MLX, ported from the PyTorch reference
//! (hexgrad/kokoro, Apache-2.0): phonemes and a voice in, 24 kHz audio out.
//!
//! The model is StyleTTS2 with an iSTFTNet vocoder. Turning text into the
//! phonemes it reads is the `astronomical-english-phonemes` crate's job.

mod acoustic_decoder;
mod albert_encoder;
mod artifact_files;
mod bidirectional_lstm;
mod configuration;
mod convolutions;
mod engine;
mod frame_alignment;
mod harmonic_source;
mod prosody_predictor;
mod reference_trace;
mod short_time_fourier;
mod snake_residual_block;
mod speech_model;
mod style_modulation;
mod style_residual_block;
mod text_encoder;
mod utterance;
mod voice_pack;
mod waveform_generator;

pub use engine::KokoroSpeechEngine;
pub use reference_trace::KokoroSpeechTrace;
