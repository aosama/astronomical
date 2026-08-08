use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration;

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation, Qwen3_5ArtifactValidator,
    Qwen3_5Model, Qwen3_5Tokenizer,
    qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure,
};

use super::Qwen3_5EngineState;
use super::ValidatedQwen3_5Artifact;

impl Qwen3_5EngineState {
    pub(super) fn speculative_prefill_draft_model_payload_bytes(&self) -> u64 {
        self.speculative_prefill_draft_model
            .as_ref()
            .map_or(0, |draft_model| {
                draft_model
                    .resident_model_payload_byte_count()
                    .saturating_add(draft_model.vision_model().map_or(0, |draft_vision_model| {
                        draft_vision_model.resident_payload_bytes()
                    }))
            })
    }

    pub(super) fn speculative_prefill_draft_maximum_expert_page_reservation_bytes(&self) -> usize {
        self.speculative_prefill_draft_model
            .as_ref()
            .and_then(|draft_model| draft_model.expert_pager.as_ref())
            .map_or(0, |expert_pager| {
                usize::try_from(expert_pager.maximum_expert_page_bytes()).unwrap_or(usize::MAX)
            })
    }

    pub(super) fn speculative_prefill_draft_supports_processed_visual_images(&self) -> bool {
        self.speculative_prefill_draft_supports_processed_visual_images
    }

    /// Releases every retained target-expert page, then materializes a draft
    /// whose MLX ownership ends when the request's prompt selection completes.
    pub(super) fn load_request_scoped_speculative_prefill_draft_model(
        &self,
        request_id: u64,
        draft_maximum_output_tokens: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5Model, InferenceEngineError> {
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "Qwen3.5 engine lost its loaded target model".to_owned(),
            })?;
        let target_expert_payload_bytes_before = target_model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        if target_expert_payload_bytes_before > 0 {
            reclaim_retained_experts_for_request_memory_pressure(
                target_model,
                usize::try_from(target_expert_payload_bytes_before).unwrap_or(usize::MAX),
            )?
            .ok_or_else(|| InferenceEngineError::InvalidRequest {
                reason: "speculative-prefill draft loading cannot evict pageable target experts"
                    .to_owned(),
            })?;
        }
        target_model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(|runtime_error| InferenceEngineError::Fatal {
                reason: runtime_error.to_string(),
            })?;
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

pub(super) fn token_identifier_mapping_digest(
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

pub(super) fn load_speculative_prefill_draft_model(
    target_model: &Qwen3_5Model,
    speculative_prefill: &WorkerSpeculativePrefillConfiguration,
    target_token_identifier_mapping_digest: [u8; 32],
    target_max_output_tokens: u32,
    memory_limits: MlxMemoryLimits,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<(Qwen3_5Model, String)>, Option<String>), InferenceEngineError> {
    if !speculative_prefill.enabled {
        return Ok((None, None));
    }
    let Some(draft_model_directory) = speculative_prefill.draft_model_directory.as_ref() else {
        return Err(configured_draft_loading_failure(
            "draft model directory resolution",
        ));
    };
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
    let draft_model = match Qwen3_5Model::load_with_performance_attribution(
        draft_runtime,
        draft_validated_artifact,
        draft_model_directory,
        false,
        true,
        performance_attribution,
    ) {
        Ok(draft_model) => match draft_model.materialize_target_weights() {
            Ok(()) => draft_model,
            Err(draft_materialization_error) => {
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
    Ok((Some((draft_model, draft_model_revision)), None))
}

fn configured_draft_loading_failure(failure_stage: &'static str) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: format!(
            "configured SpecPrefill failed during {failure_stage}; model use was stopped"
        ),
    }
}
