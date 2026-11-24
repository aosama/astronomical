//! Owner-thread folded-ternary execution state vocabulary.
//!
//! The execution structs and the pending-startup constructor live here; the
//! `MlxInferenceExecution` engine implementation that drives them lives in
//! `execution::engine_trait_impl`.

mod active_generation;
mod active_memory_breakdown;
mod attribution;
mod decode_hooks;
mod engine_trait_impl;
mod input_injection;
mod memory_limit;
mod prefill_chunk;
mod request_checks;
mod starting_decoder_state;

use crate::prism_hadamard_qwen35::artifacts::ValidatedFoldedPack;
use crate::prism_hadamard_qwen35::astro_decoder_state_cache_layout::PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::shared_execution::execution_core::{
    Qwen3_5ExecutionCore, Qwen3_5ExecutionCoreOwner, Qwen3_5ValidatedExecutionCoreSettings,
};

pub use active_generation::PrismRequestExtension;

/// Everything `load` needs, captured on the caller thread before the owner
/// thread starts: the validated shared settings and the knobs only the
/// folded-ternary load reads.
pub struct PrismHadamardQwen35PendingStartup {
    pub validated_execution_core_settings:
        Qwen3_5ValidatedExecutionCoreSettings<ValidatedFoldedPack>,
    pub pending_load: PrismHadamardQwen35PendingLoad,
}

/// The startup values only the folded-ternary load reads.
pub struct PrismHadamardQwen35PendingLoad {
    /// The ceiling and `mlx_buffer_cache` limit the runtime starts with. They
    /// stay apart from the core's current limits because `load` has always
    /// started the runtime at the worker's startup values.
    pub effective_mlx_memory_ceiling_bytes: usize,
    pub mlx_buffer_cache_limit_bytes: usize,
    pub configured_astro_ssd_prompt_cache_block_token_count: Option<usize>,
    pub ssd_prompt_cache_common_prefix_stride_blocks: u32,
    /// Lets `load` explain that this engine has no Neural Engine path.
    pub neural_engine_mlp_prefill_requested: bool,
}

/// One folded-ternary request in the shared request record: chunked prefill
/// plus one-token decode.
pub(super) type PrismHadamardQwen35ActiveGeneration = Qwen3_5EngineRequest<PrismRequestExtension>;

/// The folded-ternary execution core: the state both Qwen 3.5 engines keep,
/// holding the Prism model, request extension and validated pack.
pub(super) type PrismHadamardQwen35ExecutionCore =
    Qwen3_5ExecutionCore<PrismHadamardTextModel, PrismRequestExtension, ValidatedFoldedPack>;

/// The synchronous folded-ternary execution owned by the MLX owner thread.
pub struct PrismHadamardQwen35InferenceExecution {
    pending_load: Option<PrismHadamardQwen35PendingLoad>,
    pub(super) execution_core: PrismHadamardQwen35ExecutionCore,
    /// The `mlx_buffer_cache` limit the worker was started with; a lowered
    /// ceiling caps it, and a raised one brings it back.
    pub(super) configured_mlx_buffer_cache_bytes: u64,
    pub(super) minimum_ceiling_bytes: u64,
}

impl Qwen3_5ExecutionCoreOwner for PrismHadamardQwen35InferenceExecution {
    type Model = PrismHadamardTextModel;
    type RequestExtension = PrismRequestExtension;
    type Artifact = ValidatedFoldedPack;

    fn execution_core(&self) -> &PrismHadamardQwen35ExecutionCore {
        &self.execution_core
    }

    fn execution_core_mut(&mut self) -> &mut PrismHadamardQwen35ExecutionCore {
        &mut self.execution_core
    }
}

impl PrismHadamardQwen35InferenceExecution {
    pub fn pending(pending_startup: PrismHadamardQwen35PendingStartup) -> Self {
        let PrismHadamardQwen35PendingStartup {
            validated_execution_core_settings,
            pending_load,
        } = pending_startup;
        let mut execution_core = Qwen3_5ExecutionCore::new(validated_execution_core_settings);
        keep_prism_derived_limits(&mut execution_core);
        Self {
            configured_mlx_buffer_cache_bytes: pending_load.mlx_buffer_cache_limit_bytes as u64,
            pending_load: Some(pending_load),
            execution_core,
            minimum_ceiling_bytes: 1,
        }
    }

    /// The context window requests are held to, in the u32 form the
    /// folded-ternary checks take; it was resolved from a u32 at startup.
    pub(super) fn maximum_context_tokens(&self) -> u32 {
        u32::try_from(self.execution_core.maximum_position_count).unwrap_or(u32::MAX)
    }

    /// The prompt tokens one prefill chunk may read at most.
    pub(super) fn prompt_processing_chunk_tokens(&self) -> usize {
        self.execution_core
            .prompt_processing_chunk_sizer
            .maximum_prompt_processing_chunk_size_tokens()
    }
}

/// Replaces the three limits the shared settings derive differently from
/// what the folded-ternary engine has always used, so moving Prism's state
/// into the shared core changes no behaviour (issue #2473, Prism step 6):
///
/// - The configured context window is Prism's hard limit: it refuses any
///   request beyond it, where the Qwen 3.5 engines refuse only past the
///   artifact's own position range.
/// - Prism sets aside no context memory per token; the artifact's honest
///   figure would start charging the full-attention keys and values.
/// - Prism's key-value slabs grow in its own fixed step, whatever the
///   worker's chunking configuration asks of the Qwen 3.5 engines.
pub(super) fn keep_prism_derived_limits(execution_core: &mut PrismHadamardQwen35ExecutionCore) {
    execution_core.hard_maximum_position_count = execution_core.maximum_position_count;
    execution_core.context_memory_reservation_bytes_per_token = 0;
    execution_core.full_attention_kv_state_growth_tokens =
        PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS as i32;
}
