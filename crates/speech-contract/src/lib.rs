//! The text-to-speech contract: what a speech request is, what may go wrong
//! with one, and how its samples become audio bytes. Every side of a speech
//! request (the CLI and REST surface that accept it, the daemon that routes
//! it, the worker that speaks it) shares these definitions.

mod audio_format;
mod pcm16;
mod speech_failure;
mod speech_request;
mod wav_container;

pub use audio_format::AudioFormat;
pub use pcm16::pcm16_bytes;
pub use speech_failure::SpeechFailure;
pub use speech_request::{MAXIMUM_SPEECH_INPUT_CHARACTERS, SpeechRequest};
pub use wav_container::{WavEncodingError, wav_bytes};
