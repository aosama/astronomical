//! Memory admission for a prompt-processing chunk.
//!
//! Thin wrapper around `measure_adaptive_ram_growth_memory_admission` that
//! logs slow admissions and records expert-byte reclamation diagnostics.

use super::engine_request::Qwen3_5EngineRequest;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_engine_error};

use crate::AdaptiveRamGrowthContext;

/// Outcome of memory admission before the forward pass.
pub(super) struct AdmissionOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) retained_expert_payload_bytes_before_growth: u64,
    pub(super) target_expert_payload_bytes_reclaimed_during_context_admission: u64,
}

impl Qwen3_5EngineState {
    /// Run adaptive RAM growth admission and record reclaimed-expert diagnostics.
    ///
    /// Returns the memory snapshot before growth and the expert bytes reclaimed
    /// during context admission. The model is reborrowed after admission because
    /// the mutable admission call can trigger a Resident → Paged transition.
    pub(super) fn execute_prefill_memory_admission(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
        direct_publication_workspace_bytes: usize,
    ) -> Result<AdmissionOutcome, PromptPrefillChunkAttemptError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let target_expert_payload_bytes_before_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;

        // Admission has mutable access because this exact chunk can trigger a
        // Resident -> Paged transition. Reborrow the model afterward so graph
        // construction cannot accidentally retain the pre-transition owner.
        let admission_started_at = std::time::Instant::now();
        let (active_memory_bytes_before_growth, retained_expert_payload_bytes_before_growth) = self
            .measure_adaptive_ram_growth_memory_admission(
                adaptive_ram_growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                additional_persistent_state_growth_bytes,
                exact_temporary_workspace_bytes,
            )?;
        let admission_elapsed = admission_started_at.elapsed();
        if admission_elapsed > std::time::Duration::from_millis(500) {
            tracing::info!(
                admission_elapsed_millis = admission_elapsed.as_millis(),
                "slow admission before prefill chunk"
            );
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let target_expert_payload_bytes_after_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;

        let target_expert_payload_bytes_reclaimed_during_context_admission =
            target_expert_payload_bytes_before_context_admission
                .saturating_sub(target_expert_payload_bytes_after_context_admission);

        if direct_publication_workspace_bytes > 0
            && let Some(persistent_prompt_cache_diagnostics) =
                active_request.persistent_prompt_cache_diagnostics.as_mut()
        {
            persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_publication =
                persistent_prompt_cache_diagnostics
                    .expert_bytes_reclaimed_for_publication
                    .saturating_add(target_expert_payload_bytes_reclaimed_during_context_admission);
        }

        Ok(AdmissionOutcome {
            active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth,
            target_expert_payload_bytes_reclaimed_during_context_admission,
        })
    }
}
