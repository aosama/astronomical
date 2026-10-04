//! Compiled multi-token-prediction (MTP) verification window execution.
//!
//! A loaded model lazily compiles one trace for each supported verification
//! row count. Model-serving assembles state and weight inputs, applies the
//! trace, installs successor state through the existing owners, and falls back
//! to eager verification when the compiled lane declines.
//!
//! Design contract for the compiled window:
//!
//! 1. Token identifiers, positions, every layer's state leaves, and every
//!    quantized weight tensor enter in one frozen positional input order. The
//!    MLX-C closure ABI carries no capture context.
//! 2. Outputs: all-position logits plus every state leaf's successor, in the
//!    same fixed order. The caller installs returned states into the live
//!    request state through the existing state owners. Graph builders publish
//!    their outputs in ONE
//!    `set_graph_output_vector` call at the end: compiled-graph tracing hands
//!    builders an output vector whose context can still be null, which MLX's
//!    append operation rejects while its whole-vector set handles by
//!    allocating (measured 2026-10-03, two-output contract graph).
//! 3. One compiled graph per row count (two through four), traced on first
//!    apply and replayed thereafter. Row count and dtypes are the only shape
//!    variance the window has.
//! 4. Preserve Gated Delta boundary snapshots: verifier-prefix rollback
//!    restores convolution and recurrent state from these per-row outputs.
//!
//! The unsafe graph-builder half of this project lives in runtime-integration
//! (`mlx_compiled_verify_window_geometry` / `_graph` / `_ops`) because this
//! crate forbids unsafe code; this module owns the safe halves: assembling the
//! frozen input vector from live weights and state leaves, reading the
//! ordered outputs, and installing state successors.

pub(crate) mod compiled_window;
pub(crate) mod window_abi_assembly;
pub(crate) mod window_state_leaves;
