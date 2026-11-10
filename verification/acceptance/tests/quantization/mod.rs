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
//! - `python_parity`: against the artifacts OptiQ itself built from the same
//!   bfloat16 source: the structural conversion byte for byte, the
//!   calibration sequences token for token, the bit map the Rust allocator
//!   derives from OptiQ's own sweep, and the Rust sweep's module ranking.
//! - `mixture_of_experts_sweep`: on a synthetic mixture-of-experts model,
//!   the router and each routed-expert projection are swapped and exactly
//!   restored, and the measured sweep ranks modules like OptiQ's.

mod artifact_comparison;
mod exact_sweep;
mod mixture_of_experts_sweep;
mod parity_support;
mod python_parity;
mod requantize_throughput;
mod served_generation;
mod static_conversion;
mod support;
