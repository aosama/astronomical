//! The journeys, lowest tier first per the repository's performance-test
//! rule (start at the bottom of the call stack and move up):
//!
//! - `requantize_throughput`: the requantizer alone on the installed source,
//!   tensors per second and peak memory, with a warmup tensor first.
//! - `static_conversion`: `static` method end to end, then the artifact is
//!   loaded through Astronomical's Qwen3.5 engine and generates a Romeo and
//!   Juliet continuation with the prefill chunk fixed at 2,048 tokens.
//! - `exact_sweep`: the exact KL sweep through the model-serving probe on
//!   the installed source with two calibration sequences, checkpoint resume
//!   after a simulated interruption, then allocation and conversion.
//! - `python_parity`: an artifact the Python tool produced from the same
//!   source with the same flags, compared tensor by tensor: identical packed
//!   words at 4 bits and above (stock min/max on both sides), and squared
//!   error no worse at 2 and 3 bits; identical `per_layer` bit maps when the
//!   Python sweep's `optiq/sensitivity.json` is fed to the Rust allocator.

mod exact_sweep;
mod python_parity;
mod requantize_throughput;
mod static_conversion;
mod support;
