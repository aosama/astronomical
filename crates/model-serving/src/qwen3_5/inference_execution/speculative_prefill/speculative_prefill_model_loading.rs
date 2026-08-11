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

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration;

use crate::{
    InferenceEngineError, MlxMemoryTelemetry, PerformanceAttribution, PerformanceOperation,
    Qwen3_5ArtifactValidator, Qwen3_5Model, Qwen3_5Tokenizer,
    qwen3_5_moe::{
        Qwen3_5ExpertResidencyTransitionReason,
        reclaim_retained_experts_for_request_memory_pressure,
    },
};

use super::super::Qwen3_5EngineState;
use super::super::ValidatedQwen3_5Artifact;
use super::super::engine_request::Qwen3_5EngineRequest;

impl Qwen3_5EngineState {
    /// Captures a best-effort process-wide MLX snapshot with the live drafter
    /// represented as one logical memory category.
    ///
    /// Returning `None` is intentional telemetry behavior: inability to inspect
    /// memory must not replace the request's real scoring outcome.
    pub(crate) fn speculative_prefill_draft_memory_telemetry(
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

    /// Releases target experts, then materializes one request-scoped draft.
    ///
    /// A complete target owner is demoted as one unit; an already paged target is
    /// emptied page by page. The draft may then promote inside the capacity made
    /// available, and all of its MLX ownership ends after prompt selection.
    pub(crate) fn load_request_scoped_speculative_prefill_draft_model(
        &mut self,
        request_id: u64,
        draft_maximum_output_tokens: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5Model, InferenceEngineError> {
        // Record target retention before making room so diagnostics can attribute
        // how much expert payload draft loading displaced.
        let target_expert_payload_bytes_before = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let target_model_is_resident = self
            .model
            .as_ref()
            .is_some_and(|target_model| target_model.resident_expert_weights.is_some());
        if target_model_is_resident {
            // Resident expert ownership is all-or-nothing. Demote the complete
            // owner to paging instead of creating an unsupported partial mode.
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
        } else if target_expert_payload_bytes_before > 0 {
            // A paged target can still retain cache entries from earlier work.
            // Reclaim the full reported payload before loading another model.
            let target_model = self
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
                })?;
            reclaim_retained_experts_for_request_memory_pressure(
                target_model,
                usize::try_from(target_expert_payload_bytes_before).unwrap_or(usize::MAX),
            )?
            .ok_or_else(|| InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill draft loading cannot evict pageable target experts"
                    .to_owned(),
            })?;
        }
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?;
        target_model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(|runtime_error| InferenceEngineError::Fatal {
                reason: runtime_error.to_string(),
            })?;
        // Synchronization ensures no submitted target work still references
        // reclaimed buffers; allocator cleanup makes those bytes reusable by the
        // request-scoped draft load.
        let target_expert_payload_bytes_after = target_model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        tracing::info!(
            request_id,
            target_expert_payload_bytes_before,
            target_expert_payload_bytes_after,
            "evicted retained target experts before request-scoped speculative-prefill draft loading"
        );
        let target_token_identifier_mapping_digest = self
            .speculative_prefill_token_identifier_mapping_digest
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost the target tokenizer compatibility digest".to_owned(),
            })?;
        // Startup recorded compatibility, but the request load validates the
        // current artifact again and receives the request's output-token bound.
        let (request_scoped_draft_model, draft_unavailable_reason) = performance_attribution
            .measure_operation(
                PerformanceOperation::SpeculativePrefillRequestScopedDraftLoad,
                |performance_attribution| {
                    load_speculative_prefill_draft_model(
                        target_model,
                        &self.speculative_prefill,
                        target_token_identifier_mapping_digest,
                        draft_maximum_output_tokens,
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

/// Computes a canonical digest of tokenizer token-to-identifier semantics.
///
/// File-byte equality is intentionally insufficient: two valid tokenizer files
/// may serialize the same mapping differently. Conversely, equal vocabulary
/// sizes do not prove that an identifier means the same token in both models.
pub(crate) fn token_identifier_mapping_digest(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<[u8; 32], InferenceEngineError> {
    let tokenizer_bytes =
        validated_artifact
            .tokenizer_bytes()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "validated artifact has no tokenizer contract bytes".to_owned(),
            })?;
    Qwen3_5Tokenizer::token_identifier_mapping_digest(tokenizer_bytes).map_err(
        |tokenizer_mapping_error| InferenceEngineError::Fatal {
            reason: format!(
                "validated tokenizer mapping could not be decoded: {tokenizer_mapping_error}"
            ),
        },
    )
}

/// Loads one compatible drafter or returns a typed activation failure.
///
/// The tuple shape separates an available `(model, revision)` from an optional
/// unavailable reason retained for worker status. Configured incompatibility is
/// currently fail-closed, so validation failures return `Err` rather than a
/// silent target-only `Ok((None, reason))`.
pub(crate) fn load_speculative_prefill_draft_model(
    target_model: &Qwen3_5Model,
    speculative_prefill: &WorkerSpeculativePrefillConfiguration,
    target_token_identifier_mapping_digest: [u8; 32],
    target_max_output_tokens: u32,
    memory_limits: MlxMemoryLimits,
    should_attempt_complete_expert_residency: bool,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<(Qwen3_5Model, String)>, Option<String>), InferenceEngineError> {
    // Disabled mode performs no draft-directory access, artifact validation, or
    // MLX initialization.
    if !speculative_prefill.enabled {
        return Ok((None, None));
    }
    let Some(draft_model_directory) = speculative_prefill.draft_model_directory.as_ref() else {
        return Err(configured_draft_loading_failure(
            "draft model directory resolution",
        ));
    };
    // Validate bounded artifact structure before constructing an MLX runtime or
    // allocating model tensors.
    let draft_validated_artifact = match Qwen3_5ArtifactValidator::new()
        .validate(draft_model_directory, target_max_output_tokens)
    {
        Ok(draft_validated_artifact) => draft_validated_artifact,
        Err(draft_artifact_validation_error) => {
            tracing::warn!(
                error = %draft_artifact_validation_error,
                "configured SpecPrefill draft artifact validation failed; stopping model loading"
            );
            return Err(configured_draft_loading_failure(
                "draft model artifact validation",
            ));
        }
    };
    // Configuration identity and artifact identity must agree. This prevents a
    // directory change from silently activating a different configured model.
    if speculative_prefill.draft_model_id.as_deref() != Some(draft_validated_artifact.model_id()) {
        tracing::warn!(
            configured_draft_model_id = ?speculative_prefill.draft_model_id,
            artifact_draft_model_id = draft_validated_artifact.model_id(),
            "configured SpecPrefill draft artifact identity differs from configuration; stopping model loading"
        );
        return Err(configured_draft_loading_failure(
            "draft model identity validation",
        ));
    }
    let draft_token_identifier_mapping_digest = match token_identifier_mapping_digest(
        &draft_validated_artifact,
    ) {
        Ok(draft_token_identifier_mapping_digest) => draft_token_identifier_mapping_digest,
        Err(draft_tokenizer_contract_error) => {
            tracing::warn!(
                error = %draft_tokenizer_contract_error,
                "configured SpecPrefill draft tokenizer contract is unavailable; stopping model loading"
            );
            return Err(configured_draft_loading_failure(
                "draft tokenizer contract validation",
            ));
        }
    };
    // Semantic tokenizer equality is mandatory because the drafter's selected
    // positions are later interpreted using target token identifiers.
    if draft_token_identifier_mapping_digest != target_token_identifier_mapping_digest {
        tracing::warn!(
            "configured SpecPrefill draft and target tokenizers do not have the same contract; stopping model loading"
        );
        return Err(configured_draft_loading_failure(
            "target-and-drafter tokenizer compatibility validation",
        ));
    }
    if draft_validated_artifact.config().vocabulary_size()
        != target_model.config().vocabulary_size()
    {
        // Digest equality is the strong semantic contract; vocabulary equality
        // is retained as an explicit shape contract for output-score operations.
        tracing::warn!(
            draft_vocabulary_size = draft_validated_artifact.config().vocabulary_size(),
            target_vocabulary_size = target_model.config().vocabulary_size(),
            "configured SpecPrefill draft and target vocabularies differ; stopping model loading"
        );
        return Err(configured_draft_loading_failure(
            "target-and-drafter vocabulary compatibility validation",
        ));
    }
    let draft_model_revision = draft_validated_artifact.revision().to_owned();
    // The runtime receives the same resolved machine/user limits as the target;
    // request-scoped loading must not invent a second memory ceiling.
    let draft_runtime = match performance_attribution.measure_operation(
        PerformanceOperation::MlxRuntimeInitialization,
        |_performance_attribution| MlxRuntime::initialize(memory_limits),
    ) {
        Ok(draft_runtime) => draft_runtime,
        Err(draft_runtime_initialization_error) => {
            tracing::warn!(
                error = %draft_runtime_initialization_error,
                "configured SpecPrefill draft runtime initialization failed; stopping model loading"
            );
            return Err(configured_draft_loading_failure(
                "draft MLX runtime initialization",
            ));
        }
    };
    let mut draft_model = match Qwen3_5Model::load_with_performance_attribution(
        draft_runtime,
        draft_validated_artifact,
        draft_model_directory,
        false,
        true,
        target_model.chunking,
        performance_attribution,
    ) {
        Ok(draft_model) => match draft_model.materialize_target_weights() {
            // Materialize lazy core weights now so a nominally loaded but invalid
            // draft cannot survive startup compatibility validation.
            Ok(()) => draft_model,
            Err(draft_materialization_error) => {
                // Cleanup is best effort because the materialization error is the
                // causal failure that must be returned.
                if let Err(draft_allocator_cleanup_error) = draft_model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_allocator_cache()
                {
                    tracing::debug!(
                        error = %draft_allocator_cleanup_error,
                        "speculative prefill draft allocator cleanup failed after materialization failure"
                    );
                }
                tracing::warn!(
                    error = %draft_materialization_error,
                    "configured SpecPrefill draft weights could not be materialized; stopping model loading"
                );
                return Err(configured_draft_loading_failure(
                    "draft weight materialization",
                ));
            }
        },
        Err(draft_load_error) => {
            // MLX allocation is process-global; clear any partial draft load from
            // the shared runtime before reporting activation failure.
            if let Err(draft_allocator_cleanup_error) = target_model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache()
            {
                tracing::debug!(
                    error = %draft_allocator_cleanup_error,
                    "speculative prefill draft allocator cleanup failed after load failure"
                );
            }
            tracing::warn!(
                error = %draft_load_error,
                "configured SpecPrefill draft model could not be loaded; stopping model loading"
            );
            return Err(configured_draft_loading_failure("draft weight loading"));
        }
    };
    // Startup loads the draft only to validate compatibility and drops it before
    // target admission, so resident materialization there would be wasted work.
    // The request-scoped load executes scoring and can benefit from residency.
    if should_attempt_complete_expert_residency {
        // Only request-scoped loads execute scoring. Startup validation skips
        // promotion because the draft is immediately dropped afterward.
        draft_model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(|draft_cleanup_error| {
                tracing::warn!(
                    error = %draft_cleanup_error,
                    "request-scoped speculative prefill draft cleanup failed before residency admission"
                );
                configured_draft_loading_failure("draft expert residency cleanup")
            })?;
        draft_model
            .try_promote_experts_to_resident(
                Qwen3_5ExpertResidencyTransitionReason::SpeculativePrefillDraftLoading,
                performance_attribution,
            )
            .map_err(|draft_residency_error| {
                tracing::warn!(
                    error = %draft_residency_error,
                    "request-scoped speculative prefill draft expert residency failed"
                );
                configured_draft_loading_failure("draft expert residency")
            })?;
    }
    Ok((Some((draft_model, draft_model_revision)), None))
}

/// Produces the one bounded fatal error used for configured drafter activation.
fn configured_draft_loading_failure(failure_stage: &'static str) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: format!(
            "configured SpecPrefill failed during {failure_stage}; model use was stopped"
        ),
    }
}
