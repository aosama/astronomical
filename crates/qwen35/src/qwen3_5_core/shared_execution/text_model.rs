//! What the shared execution steps ask of a Qwen3.5 text model.
//!
//! The resident and streaming models differ in how they hold expert weights,
//! so each answers these questions its own way; the steps that ask them are
//! shared. The trait grows as more of the generation loop becomes shared.

use ipc_protocol::ExpertMemoryMode;
use memory_policy::ledger::LedgerReading;

use crate::PerformanceAttribution;
use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

pub trait Qwen3_5TextModel: Qwen3_5MemoryOwnership {
    /// How expert weights are held as a reply token is emitted. The resident
    /// model always holds them whole; the streaming model reports what it
    /// observed for the forward that produced the token.
    fn observed_expert_memory_mode(&self) -> ExpertMemoryMode;

    /// How the model holds expert weights by ownership, before any forward
    /// has given evidence; a request's start reports this.
    fn expert_memory_mode(&self) -> ExpertMemoryMode;

    /// The expert residency claim derived from one memory breakdown, so the
    /// telemetry and the claim describe the same instant.
    fn expert_residency_telemetry_for_breakdown(
        &self,
        active_memory_breakdown: &MlxActiveMemoryBreakdown,
    ) -> ExpertResidencyTelemetry;

    /// The memory governor's latest ledger reading, when it has one.
    fn latest_memory_ledger_reading(&self) -> Option<LedgerReading>;

    /// The MLX memory reading telemetry reports. The streaming model raises
    /// its peak to the largest prompt layer it measured, which MLX's own
    /// counter misses when layers are evaluated one at a time.
    fn memory_snapshot_for_telemetry(
        &self,
    ) -> Result<runtime_integration::MlxMemorySnapshot, crate::InferenceEngineError>;

    /// Reads a stretch of tokens into the request's decoder state without
    /// sampling from them.
    fn prefill_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError>;

    /// Builds the forward of a stretch of tokens and returns its last
    /// token's logits, still lazy.
    fn build_forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<mlx_bridge::MlxArray, Qwen3_5ExecutionError>;

    /// Submits a sampled token and the decoder state for evaluation on the
    /// graphics processor without waiting for it.
    fn async_evaluate_generation(
        &self,
        generated_token: &mlx_bridge::MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5ExecutionError>;
}
