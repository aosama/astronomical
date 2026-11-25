//! Owner-thread folded-ternary execution state vocabulary.
//!
//! The execution structs and the pending-startup constructor live here; the
//! `MlxInferenceExecution` engine implementation that drives them lives in
//! `execution::engine_trait_impl`.

mod active_generation;
mod astro_ssd_prompt_cache_hooks;
mod decode_hooks;
mod engine_trait_impl;
mod input_injection_hooks;
mod memory_limit;
mod model_loading_hooks;
mod prefill_chunk;
mod starting_decoder_state;

use crate::prism_hadamard_qwen35::artifacts::ValidatedFoldedPack;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use text_generation::engine_request::TextEngineRequest;
use text_generation::generation_loop::execution_core::{
    TextExecutionCore, TextExecutionCoreOwner, TextValidatedExecutionCoreSettings,
};

pub use active_generation::PrismRequestExtension;

/// Everything the engine needs, captured on the caller thread before the
/// owner thread starts: the validated shared settings and the one load knob
/// only the folded-ternary engine reads.
pub struct PrismHadamardQwen35PendingStartup {
    pub validated_execution_core_settings: TextValidatedExecutionCoreSettings<ValidatedFoldedPack>,
    /// Lets `load` explain that this engine has no Neural Engine path.
    pub neural_engine_mlp_prefill_requested: bool,
}

/// One folded-ternary request in the shared request record: chunked prefill
/// plus one-token decode.
pub(super) type PrismHadamardQwen35ActiveGeneration = TextEngineRequest<PrismRequestExtension>;

/// The folded-ternary execution core: the state both Qwen 3.5 engines keep,
/// holding the Prism model, request extension and validated pack.
pub(super) type PrismHadamardQwen35ExecutionCore =
    TextExecutionCore<PrismHadamardTextModel, PrismRequestExtension, ValidatedFoldedPack>;

/// The synchronous folded-ternary execution owned by the MLX owner thread.
pub struct PrismHadamardQwen35InferenceExecution {
    pub(super) execution_core: PrismHadamardQwen35ExecutionCore,
    /// The `mlx_buffer_cache` limit the worker was started with; a lowered
    /// ceiling caps it, and a raised one brings it back.
    pub(super) configured_mlx_buffer_cache_bytes: u64,
    pub(super) neural_engine_mlp_prefill_requested: bool,
}

impl TextExecutionCoreOwner for PrismHadamardQwen35InferenceExecution {
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
            neural_engine_mlp_prefill_requested,
        } = pending_startup;
        let execution_core = TextExecutionCore::new(validated_execution_core_settings);
        Self {
            configured_mlx_buffer_cache_bytes: execution_core
                .memory_limits
                .mlx_buffer_cache_limit_bytes()
                as u64,
            execution_core,
            neural_engine_mlp_prefill_requested,
        }
    }
}
