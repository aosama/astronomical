//! The Prism text model as the shared Qwen 3.5 execution steps ask for it
//! (issue #2473, Prism step 5): its memory ownership and its forwards.
//!
//! A folded pack is dense and fully resident: it holds no experts, keeps its
//! whole weight payload in memory, and has no memory governor. Every answer
//! here repeats what the Prism engine computes or reports today, and every
//! forward delegates to the same Prism forward and evaluation code, so the
//! shared loop would build and submit the same graphs Prism's own loop does.

use ipc_protocol::ExpertMemoryMode;
use memory_policy::ledger::LedgerReading;
use mlx_bridge::MlxArray;
use runtime_integration::MlxMemorySnapshot;

use super::{PrismHadamardTextModel, request_decoder_state_evaluation_arrays};
use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::memory::MemoryReserveSnapshot;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};
use crate::{RequestDecoderStateStack, memory};
use text_generation::engine_errors::fatal_runtime_error;
use text_generation::execution_error::TextExecutionError;
use text_generation::model::memory_breakdown::TextModelMemoryOwnership;
use text_generation::model::text_model::TextModel;

impl TextModelMemoryOwnership for PrismHadamardTextModel {
    /// The ceiling MLX enforces right now. The Prism engine applies every
    /// ceiling change to this runtime, so the native limit is the engine's
    /// ceiling; an unreadable limit reads as zero, leaving no headroom to
    /// explain rather than inventing one.
    fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.runtime
            .configured_memory_limit_bytes()
            .map_or(0, usize_to_u64_saturating)
    }

    fn held_expert_payload_bytes(&self) -> u64 {
        0
    }

    /// The whole pack, as the Prism engine's memory breakdown charges it.
    fn held_model_core_payload_bytes(&self) -> u64 {
        self.pack.total_payload_bytes()
    }

    /// No memory governor sets Prism's reserves aside.
    fn governed_memory_reserve_snapshot(&self) -> Option<MemoryReserveSnapshot> {
        None
    }
}

// Named calls: method syntax inside these bodies would resolve to the trait
// itself if an inherent method were ever renamed, recursing forever.
impl TextModel for PrismHadamardTextModel {
    fn observed_expert_memory_mode(&self) -> ExpertMemoryMode {
        ExpertMemoryMode::Resident
    }

    fn expert_memory_mode(&self) -> ExpertMemoryMode {
        ExpertMemoryMode::Resident
    }

    /// A dense pack has no experts: nothing is seated, nothing is measured.
    fn expert_residency_telemetry_for_breakdown(
        &self,
        _active_memory_breakdown: &MlxActiveMemoryBreakdown,
    ) -> ExpertResidencyTelemetry {
        ExpertResidencyTelemetry::default()
    }

    fn latest_memory_ledger_reading(&self) -> Option<LedgerReading> {
        None
    }

    fn memory_snapshot_for_telemetry(&self) -> Result<MlxMemorySnapshot, InferenceEngineError> {
        self.runtime.memory_snapshot().map_err(fatal_runtime_error)
    }

    /// The same floor the Prism engine computes at load.
    fn minimum_mlx_memory_ceiling_bytes(&self) -> Result<u64, InferenceEngineError> {
        Ok(
            memory::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(
                self.pack.total_payload_bytes(),
            ),
        )
    }

    fn resident_model_payload_byte_count(&self) -> u64 {
        self.pack.total_payload_bytes()
    }

    /// Runs the chunk's forward and evaluates its final hidden row, the same
    /// two measured regions and the same single evaluation root the Prism
    /// engine uses for a prompt chunk that does not end the prompt.
    fn prefill_chunk_with_performance_attribution(
        &self,
        prompt_chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let token_indices = self.token_indices(prompt_chunk_token_ids)?;
        let final_hidden_state_row = performance_attribution.measure_operation(
            PerformanceOperation::PromptPrefillAdvanceSpan,
            |_performance_attribution| {
                PrismHadamardTextModel::forward_hidden_states(
                    self,
                    &token_indices,
                    starting_position_tokens,
                    request_decoder_state,
                )
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
            |_performance_attribution| self.runtime.evaluate_arrays(&[&final_hidden_state_row]),
        )?;
        Ok(())
    }

    /// The chunk's final-position logits, still lazy. Prism's forward has no
    /// finer measured regions, so nothing is recorded here; the caller
    /// measures the step around it.
    fn build_forward_chunk_with_performance_attribution(
        &self,
        forwarded_token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        _performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let token_indices = self.token_indices(forwarded_token_ids)?;
        Ok(PrismHadamardTextModel::forward_pass(
            self,
            &token_indices,
            starting_position_tokens,
            request_decoder_state,
        )?)
    }

    /// Submits the sampled token with the same evaluation roots, in the same
    /// order, as the Prism engine's decode step.
    fn async_evaluate_generation(
        &self,
        sampled_generated_token: &MlxArray,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), TextExecutionError> {
        self.runtime
            .async_eval_arrays(&request_decoder_state_evaluation_arrays(
                request_decoder_state,
                sampled_generated_token,
            ))?;
        Ok(())
    }
}

impl PrismHadamardTextModel {
    /// The `[1, token_count]` index array of `token_ids`. Each id keeps its
    /// bits, as the Prism engine's own conversion does; the serving checks
    /// already hold every id inside the vocabulary.
    fn token_indices(&self, token_ids: &[u32]) -> Result<MlxArray, TextExecutionError> {
        let signed_token_ids: Vec<i32> = token_ids.iter().copied().map(u32::cast_signed).collect();
        Ok(self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?)
    }
}

fn usize_to_u64_saturating(byte_count: usize) -> u64 {
    u64::try_from(byte_count).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "text_model_tests.rs"]
mod text_model_tests;
