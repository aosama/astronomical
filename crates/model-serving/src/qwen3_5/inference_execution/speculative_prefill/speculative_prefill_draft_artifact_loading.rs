//! Validates and materializes one configured SpecPrefill drafter artifact.

use astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation, Qwen3_5ArtifactValidator,
    Qwen3_5Model, Qwen3_5Tokenizer,
};

use super::super::ValidatedQwen3_5Artifact;

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
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<(Qwen3_5Model, String)>, Option<String>), InferenceEngineError> {
    // Disabled mode performs no draft-directory access, artifact validation, or
    // MLX initialization.
    let Some(draft_validated_artifact) = validate_speculative_prefill_draft_artifact(
        target_model,
        speculative_prefill,
        target_token_identifier_mapping_digest,
        target_max_output_tokens,
    )?
    else {
        return Ok((None, None));
    };
    load_validated_speculative_prefill_draft_model(
        target_model,
        speculative_prefill,
        draft_validated_artifact,
        memory_limits,
        performance_attribution,
    )
}

pub(crate) fn validate_speculative_prefill_draft_artifact(
    target_model: &Qwen3_5Model,
    speculative_prefill: &WorkerSpeculativePrefillConfiguration,
    target_token_identifier_mapping_digest: [u8; 32],
    target_max_output_tokens: u32,
) -> Result<Option<ValidatedQwen3_5Artifact>, InferenceEngineError> {
    if !speculative_prefill.enabled {
        return Ok(None);
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
    Ok(Some(draft_validated_artifact))
}

pub(crate) fn load_validated_speculative_prefill_draft_model(
    target_model: &Qwen3_5Model,
    speculative_prefill: &WorkerSpeculativePrefillConfiguration,
    draft_validated_artifact: ValidatedQwen3_5Artifact,
    memory_limits: MlxMemoryLimits,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(Option<(Qwen3_5Model, String)>, Option<String>), InferenceEngineError> {
    let draft_model_directory = speculative_prefill
        .draft_model_directory
        .as_ref()
        .ok_or_else(|| configured_draft_loading_failure("draft model directory resolution"))?;
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
    let draft_model = match Qwen3_5Model::load_with_performance_attribution(
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
