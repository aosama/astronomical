//! The folded ternary family's performance tiers, climbing the call stack:
//!
//! 1. `operation_throughput`: the Hadamard activation transform and the
//!    folded ternary matmul alone, at the pack's real shapes.
//! 2. `layer_throughput`: one gated-delta and one full-attention decoder
//!    layer, prefill-shaped and decode-shaped.
//! 3. `forward_throughput`: whole-model prefill chunks and decode steps
//!    through the raw forward pass.
//! 4. `generation_throughput`: the greedy generation session, whose decode
//!    loop is the path the serving engine mirrors.
//!
//! Every tier loads the real pack, touches the graphics processor, and is
//! therefore `#[serial]` and `#[ignore]`d; run them through
//! `cargo test-prism-hadamard-performance` (an optimized build) with
//! `model_directory` selected in the explicit test-run configuration.

pub(crate) mod support;

mod elementwise_fusion_parity;
mod forward_throughput;
mod generation_throughput;
mod layer_throughput;
mod operation_throughput;
