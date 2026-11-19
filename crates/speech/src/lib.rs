//! Text-to-speech for Astronomical (issue #2340): the worker-facing seam
//! every speech family implements, and the Kokoro and Phonon families.
//!
//! The families sit beside the other model families rather than inside the
//! serving crate, so a speech change rebuilds and retests only this crate.
//! The edge is one way: this crate never depends on
//! `astronomical-model-serving`, which `cargo xtask check-architecture`
//! enforces; the worker there drives these engines through the seam.

pub mod speech_engine;

#[cfg(feature = "direct-mlx")]
pub mod speech_models;
