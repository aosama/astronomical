//! Validates and materializes the optional Qwen3.5 drafter model.
//!
//! Startup performs a short-lived load to prove artifact, tokenizer, vocabulary,
//! runtime, and weight compatibility. That model is dropped before target expert
//! residency admission. A later eligible request repeats the validated load as a
//! request-scoped owner, optionally promotes its experts, scores the prompt, and
//! drops every drafter allocation before target execution resumes.
//!
//! This deliberate reload trades load work for a much smaller steady-state
//! footprint: the target and complete drafter are never permanent co-residents.

use crate::{
    InferenceEngineError, MlxMemoryTelemetry, PerformanceAttribution, PerformanceOperation,
    Qwen3_5Model,
    qwen3_5_moe::{
        Qwen3_5ExpertResidencyTransitionReason,
        reclaim_retained_experts_for_request_memory_pressure,
    },
};

use super::super::Qwen3_5EngineState;
use super::super::engine_request::Qwen3_5EngineRequest;
use super::speculative_prefill_draft_artifact_loading::{
    load_validated_speculative_prefill_draft_model, validate_speculative_prefill_draft_artifact,
};
use crate::SpeculativePrefillAdmission;

impl Qwen3_5EngineState {
    /// Captures a best-effort process-wide MLX snapshot with the live drafter
    /// represented as one logical memory category.
    ///
    /// Returning `None` is intentional telemetry behavior: inability to inspect
    /// memory must not replace the request's real scoring outcome.
    pub(in crate::qwen3_5) fn speculative_prefill_draft_memory_telemetry(
        &self,
        active_request: &Qwen3_5EngineRequest,
        draft_model: &Qwen3_5Model,
        draft_request_decoder_state: &super::super::super::RequestDecoderStateStack,
        draft_visual_embeddings: Option<&astronomical_runtime_integration::MlxArray>,
    ) -> Option<MlxMemoryTelemetry> {
        let target_model = self.model.as_ref()?;
        let mlx_memory_snapshot = draft_model.runtime().memory_snapshot().ok()?;
        let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes()).ok()?;
        let allocator_cache_memory_bytes =
            u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).ok()?;
        let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes()).ok()?;
        Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            target_model.active_memory_breakdown_with_speculative_prefill_draft(
                active_request.request_decoder_state(),
                active_request.additional_context_state_payload_bytes(),
                active_memory_bytes,
                draft_model,
                draft_request_decoder_state,
                draft_visual_embeddings,
            ),
        ))
    }

    /// Returns the largest single routed expert page a paged drafter may need.
    ///
    /// Dense and fully resident models have no pager and therefore reserve zero.
    /// Conversion overflow saturates so memory admission fails safely rather
    /// than under-reserving an unrepresentable page.
    pub(crate) fn speculative_prefill_draft_maximum_expert_page_reservation_bytes(&self) -> usize {
        self.speculative_prefill_draft_model
            .as_ref()
            .and_then(|draft_model| draft_model.expert_pager.as_ref())
            .map_or(0, |expert_pager| {
                usize::try_from(expert_pager.maximum_expert_page_bytes()).unwrap_or(usize::MAX)
            })
    }

    /// Reports startup-validated visual compatibility between target and drafter.
    pub(crate) fn speculative_prefill_draft_supports_processed_visual_images(&self) -> bool {
        self.speculative_prefill_draft_supports_processed_visual_images
    }

    /// Materializes one request-scoped draft beside the target when capacity permits.
    ///
    /// A complete target owner is demoted only when target plus draft artifact
    /// payload exceeds the stable ceiling. Paged targets reclaim only the required
    /// bytes. All drafter MLX ownership ends after prompt selection.
    pub(crate) fn load_request_scoped_speculative_prefill_draft_model(
        &mut self,
        request_id: u64,
        draft_maximum_output_tokens: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5Model, InferenceEngineError> {
        // Record target retention before making room so diagnostics can attribute
        // how much expert payload draft loading displaced.
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?;
        let target_expert_payload_bytes_before = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let target_model_is_resident = target_model.resident_expert_weights.is_some();
        let target_token_identifier_mapping_digest = self
            .speculative_prefill_token_identifier_mapping_digest
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost the target tokenizer compatibility digest".to_owned(),
            })?;
        let draft_validated_artifact = validate_speculative_prefill_draft_artifact(
            target_model,
            &self.speculative_prefill,
            target_token_identifier_mapping_digest,
            draft_maximum_output_tokens,
        )?
        .ok_or_else(|| InferenceEngineError::InvalidRequest {
            reason: "speculative-prefill request-scoped draft loading found no enabled drafter"
                .to_owned(),
        })?;
        let draft_artifact_payload_bytes =
            usize::try_from(draft_validated_artifact.total_payload_bytes()).map_err(|_| {
                InferenceEngineError::Fatal {
                    reason: "speculative-prefill drafter payload exceeds the platform range"
                        .to_owned(),
                }
            })?;
        let target_memory_snapshot_before_draft_load = target_model
            .runtime()
            .memory_snapshot()
            .map_err(|runtime_error| InferenceEngineError::Fatal {
                reason: runtime_error.to_string(),
            })?;
        let stable_active_memory_ceiling_bytes = self.memory_limits.active_memory_limit_bytes();
        let draft_load_fits_with_current_target =
            SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
                target_memory_snapshot_before_draft_load.active_memory_bytes(),
                draft_artifact_payload_bytes,
                stable_active_memory_ceiling_bytes,
            );
        let target_memory_snapshot_after_draft_load_preparation = if target_model_is_resident
            && !draft_load_fits_with_current_target
        {
            self.model
                .as_mut()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
                })?
                .demote_resident_experts_to_paging(
                    Qwen3_5ExpertResidencyTransitionReason::SpeculativePrefillDraftLoading,
                    performance_attribution,
                )
                .map_err(InferenceEngineError::from)?;
            self.model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
                })?
                .runtime()
                .memory_snapshot()
                .map_err(|runtime_error| InferenceEngineError::Fatal {
                    reason: runtime_error.to_string(),
                })?
        } else if !target_model_is_resident && !draft_load_fits_with_current_target {
            let target_model = self
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
                })?;
            let required_target_expert_reclamation_bytes =
                SpeculativePrefillAdmission::required_target_expert_reclamation_bytes(
                    target_memory_snapshot_before_draft_load.active_memory_bytes(),
                    draft_artifact_payload_bytes,
                    stable_active_memory_ceiling_bytes,
                );
            reclaim_retained_experts_for_request_memory_pressure(
                target_model,
                required_target_expert_reclamation_bytes,
            )?
            .ok_or_else(|| InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill draft loading cannot evict pageable target experts"
                    .to_owned(),
            })?
        } else {
            target_model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
                .map_err(|runtime_error| InferenceEngineError::Fatal {
                    reason: runtime_error.to_string(),
                })?;
            target_model
                .runtime()
                .memory_snapshot()
                .map_err(|runtime_error| InferenceEngineError::Fatal {
                    reason: runtime_error.to_string(),
                })?
        };
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?;
        // Logical expert eviction is not enough evidence: immutable in-flight
        // snapshots may have delayed physical release. Only this fresh active
        // memory sample can authorize loading the complete draft payload.
        if !SpeculativePrefillAdmission::draft_load_fits_with_target_active_memory(
            target_memory_snapshot_after_draft_load_preparation.active_memory_bytes(),
            draft_artifact_payload_bytes,
            stable_active_memory_ceiling_bytes,
        ) {
            target_model.resume_expert_retention_after_request_memory_pressure();
            return Err(InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill drafter does not fit after target expert reclamation"
                    .to_owned(),
            });
        }
        let target_expert_payload_bytes_after = target_model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        tracing::info!(
            request_id,
            target_model_is_resident,
            target_active_memory_bytes_before_draft_load =
                target_memory_snapshot_before_draft_load.active_memory_bytes(),
            target_active_memory_bytes_after_draft_load_preparation =
                target_memory_snapshot_after_draft_load_preparation.active_memory_bytes(),
            draft_artifact_payload_bytes,
            stable_active_memory_ceiling_bytes,
            draft_load_fits_with_current_target,
            target_expert_payload_bytes_before,
            target_expert_payload_bytes_after,
            "prepared target residency for request-scoped speculative-prefill draft loading"
        );
        performance_attribution.record_counter(
            crate::PerformanceCounter::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
            target_expert_payload_bytes_before.saturating_sub(target_expert_payload_bytes_after),
        );
        let (request_scoped_draft_model, draft_unavailable_reason) = performance_attribution
            .measure_operation(
                PerformanceOperation::SpeculativePrefillRequestScopedDraftLoad,
                |performance_attribution| {
                    load_validated_speculative_prefill_draft_model(
                        target_model,
                        &self.speculative_prefill,
                        draft_validated_artifact,
                        self.memory_limits,
                        true,
                        performance_attribution,
                    )
                },
            )?;
        request_scoped_draft_model
            .map(|(draft_model, _draft_model_revision)| draft_model)
            .ok_or_else(|| InferenceEngineError::InvalidRequest {
                reason: format!(
                    "speculative-prefill request-scoped draft loading failed: {}",
                    draft_unavailable_reason.unwrap_or_else(|| "no reason was reported".to_owned())
                ),
            })
    }
}
