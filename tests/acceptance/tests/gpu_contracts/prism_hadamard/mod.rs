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
//! `PRISM_HADAMARD_QWEN35_PACK_DIR` set.

pub(crate) mod support;

mod forward_throughput;
mod generation_throughput;
mod layer_throughput;
mod operation_throughput;
