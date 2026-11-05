//! Opt-in prefill that runs a slice of each full chunk's tokens through whole
//! dense MLP programs on the Neural Engine, next to the GPU (issue #161).
//!
//! Only the MLP moves: as one program it needs one hand-off per layer, where
//! splitting single projections by channel needs one per projection. Mixture-of-experts models are out of scope because their routed experts stay on
//! the GPU, which measured about 1.04× for the chunk.

mod program_loading;
mod token_split_forward;

pub(crate) use program_loading::prepare_neural_engine_mlp_prefill;

use std::cell::{Cell, RefCell};

use astronomical_neural_engine::core_ml_projection::CoreMlProjection;

// 47 percent: on a real mixed-bit artifact at 44 percent the GPU finished last in
// 88 percent of layers (18.6 ms against 16.5 ms), and each 64 tokens moves about
// 1 ms between the engines. The chunk's other half, attention and the mixer, runs
// on the GPU regardless, which is why real prefill through the daemon measured
// 1.13× to 1.2× on Ornith-1.5-9B, with the prompt cache on and off, and not more.
const NEURAL_ENGINE_SHARE_NUMERATOR: u32 = 15;
const NEURAL_ENGINE_SHARE_DENOMINATOR: u32 = 32;
// A surface row must fill its alignment exactly for MLX to read it as one array,
// and multiples of 64 fp16 tokens do.
const NEURAL_ENGINE_TOKEN_GRANULARITY: u32 = 64;

/// One program per decoder layer, each fixed to the chunk's token slice.
pub(crate) struct NeuralEngineMlpPrefill {
    layer_programs: RefCell<Vec<CoreMlProjection>>,
    chunk_token_count: u32,
    neural_engine_token_count: u32,
    /// Set by the first runtime failure, so later chunks stay on the GPU.
    has_failed: Cell<bool>,
}

impl std::fmt::Debug for NeuralEngineMlpPrefill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NeuralEngineMlpPrefill")
            .field("chunk_token_count", &self.chunk_token_count)
            .field("neural_engine_token_count", &self.neural_engine_token_count)
            .field("has_failed", &self.has_failed.get())
            .finish_non_exhaustive()
    }
}

impl NeuralEngineMlpPrefill {
    /// Programs have fixed shapes, so tails, reduced chunks, and one-token decode
    /// all stay on the GPU. Fixed-shape execution never pads tokens to fit.
    fn handles(&self, token_count: i32) -> bool {
        !self.has_failed.get() && u32::try_from(token_count) == Ok(self.chunk_token_count)
    }
}

/// How many of a chunk's tokens the Neural Engine takes, or `None` when the chunk
/// is too small for a whole multiple of the granularity.
fn neural_engine_tokens_for_chunk(chunk_token_count: u32) -> Option<u32> {
    let share_token_count = u64::from(chunk_token_count) * u64::from(NEURAL_ENGINE_SHARE_NUMERATOR)
        / u64::from(NEURAL_ENGINE_SHARE_DENOMINATOR);
    let aligned_token_count = u32::try_from(share_token_count).ok()?
        / NEURAL_ENGINE_TOKEN_GRANULARITY
        * NEURAL_ENGINE_TOKEN_GRANULARITY;
    (aligned_token_count > 0).then_some(aligned_token_count)
}

#[cfg(test)]
#[path = "neural_engine_mlp_prefill/neural_engine_mlp_prefill_tests.rs"]
mod tests;
