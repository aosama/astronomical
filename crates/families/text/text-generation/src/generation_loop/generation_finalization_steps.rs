//! The finalization steps every text engine takes: the memory telemetry a
//! finished or idle engine reports, the performance attribution record of a
//! generation, cancelling a request, and finalizing a request that ended in
//! an error.
//!
//! The order in which an engine drops its request, ends its governed request
//! and releases MLX memory differs between the engines, so the finalization
//! itself stays with each engine; these steps are what it calls.

use runtime_integration::MlxMemorySnapshot;

use crate::engine_errors::{
    mlx_active_memory_bytes_exceed_u64_range_error, mlx_buffer_cache_bytes_exceed_u64_range_error,
    mlx_peak_memory_bytes_exceed_u64_range_error,
};
use crate::engine_request::TextEngineRequest;
use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::{
    GenerationFinalization, GenerationPerformanceAttributionMetadata, InferenceEngineError,
    MlxMemoryTelemetry, PerformanceAttribution, PerformanceAttributionOutcome,
};

use crate::model::memory_breakdown::TextModelMemoryOwnership;

use super::execution_core::TextExecutionCoreOwner;
use crate::model::text_model::TextModel;

/// What an engine supplies to the shared finalization steps.
pub trait TextFinalizationHooks: TextExecutionCoreOwner<Model: TextModel> {
    /// Decoder-state bytes kept from the last reply for the next one; only
    /// the resident engine keeps any.
    fn retained_live_context_state_payload_bytes(&self) -> u64 {
        0
    }

    /// The MLX working buffers measured for one prompt chunk of the size this
    /// engine serves; `None` while nothing has measured one.
    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64>;

    /// Adds what this engine reports beyond the shared finalization; the
    /// streaming engine adds its expert streaming figures.
    fn complete_generation_finalization(
        &self,
        generation_finalization: GenerationFinalization,
        _expert_memory_mode: Option<ipc_protocol::ExpertMemoryMode>,
        _expert_residency_telemetry: Option<ExpertResidencyTelemetry>,
    ) -> GenerationFinalization {
        generation_finalization
    }

    /// Ends the request: the engine's own order of dropping, governing and
    /// releasing memory, then the record and the finalization report.
    fn finalize_generation_request(
        &mut self,
        active_request: TextEngineRequest<Self::RequestExtension>,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization;
}

/// The engine's MLX memory as the status surface shows it between requests;
/// `None` with no model loaded.
pub fn collect_current_mlx_memory_telemetry<Engine: TextFinalizationHooks>(
    engine: &Engine,
) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
    let Some(model) = engine.execution_core().model.as_ref() else {
        return Ok(None);
    };
    let mlx_memory_snapshot = model.memory_snapshot_for_telemetry()?;
    let (_expert_residency_telemetry, mlx_memory_telemetry) =
        idle_mlx_memory_telemetry_for_snapshot(engine, model, &mlx_memory_snapshot)?;
    Ok(Some(mlx_memory_telemetry))
}

/// The between-request telemetry of one memory reading, and the expert
/// residency claim derived from the same breakdown, so a report that carries
/// both cannot straddle a promotion or demote (issue #337).
pub fn idle_mlx_memory_telemetry_for_snapshot<Engine: TextFinalizationHooks>(
    engine: &Engine,
    model: &Engine::Model,
    mlx_memory_snapshot: &MlxMemorySnapshot,
) -> Result<(ExpertResidencyTelemetry, MlxMemoryTelemetry), InferenceEngineError> {
    let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(mlx_active_memory_bytes_exceed_u64_range_error)?;
    let mlx_buffer_cache_bytes = u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes())
        .map_err(mlx_buffer_cache_bytes_exceed_u64_range_error)?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(mlx_peak_memory_bytes_exceed_u64_range_error)?;
    let active_memory_breakdown = model.finalized_active_memory_breakdown(
        active_memory_bytes,
        engine.retained_live_context_state_payload_bytes(),
        0,
    );
    let expert_residency_telemetry =
        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown);
    let mlx_memory_telemetry = between_request_mlx_memory_telemetry(
        model,
        MlxMemoryCounters {
            active_memory_bytes,
            mlx_buffer_cache_bytes,
            peak_memory_bytes,
        },
        active_memory_breakdown,
    )
    .with_expert_residency_telemetry(expert_residency_telemetry);
    Ok((expert_residency_telemetry, mlx_memory_telemetry))
}

/// One MLX memory reading in the u64 units telemetry reports.
pub struct MlxMemoryCounters {
    pub active_memory_bytes: u64,
    pub mlx_buffer_cache_bytes: u64,
    pub peak_memory_bytes: u64,
}

/// The telemetry of a reading taken between requests, which the idle status
/// sample and a finished request's report both are.
pub fn between_request_mlx_memory_telemetry<Model: TextModel>(
    model: &Model,
    mlx_memory_counters: MlxMemoryCounters,
    active_memory_breakdown: MlxActiveMemoryBreakdown,
) -> MlxMemoryTelemetry {
    MlxMemoryTelemetry::new(
        mlx_memory_counters.active_memory_bytes,
        mlx_memory_counters.mlx_buffer_cache_bytes,
        mlx_memory_counters.peak_memory_bytes,
        active_memory_breakdown,
    )
    // Issue #510: a between-request sample answers "is the remaining
    // headroom free or promised" from the memory governor's accounts, and a
    // sample without the split would overwrite the load-complete split the
    // menu paints.
    .with_memory_ceiling_utilization(
        model.between_request_memory_ceiling_utilization_for_breakdown(
            mlx_memory_counters.active_memory_bytes,
            active_memory_breakdown,
        ),
    )
    .with_memory_ledger_reading(model.latest_memory_ledger_reading())
}

/// Ends the request the client stopped. A request that already ended on its
/// own was finalized by the decode step and is gone, so there is nothing to
/// report; a stop meant for another request leaves the running one alone,
/// because a late stop for a finished request must not end its successor.
pub fn cancel_generation<Engine: TextFinalizationHooks>(
    engine: &mut Engine,
    request_id: ipc_protocol::RequestId,
) -> GenerationFinalization {
    let Some(active_request) = engine.execution_core_mut().active_request.take() else {
        return GenerationFinalization::default();
    };
    if active_request.request_id != request_id {
        engine.execution_core_mut().active_request = Some(active_request);
        return GenerationFinalization::default();
    }
    engine.finalize_generation_request(
        active_request,
        PerformanceAttributionOutcome::Cancelled,
        None,
    )
}

/// Finalizes a request that ended in an error, recording it as rejected when
/// the request was refused and as failed when the engine failed.
pub fn finalize_generation_request_after_error<Engine: TextFinalizationHooks>(
    engine: &mut Engine,
    active_request: TextEngineRequest<Engine::RequestExtension>,
    generation_error: &InferenceEngineError,
    rejection_description: &'static str,
    failure_description: &'static str,
) {
    let (performance_attribution_outcome, performance_attribution_failure_description) =
        error_outcome(generation_error, rejection_description, failure_description);
    let _generation_finalization = engine.finalize_generation_request(
        active_request,
        performance_attribution_outcome,
        Some(performance_attribution_failure_description),
    );
}

/// A refused request is a rejection, with the rejection's description; an
/// engine failure is a failure, with the failure's.
pub fn error_outcome(
    generation_error: &InferenceEngineError,
    rejection_description: &'static str,
    failure_description: &'static str,
) -> (PerformanceAttributionOutcome, &'static str) {
    match generation_error {
        InferenceEngineError::InvalidRequest { .. }
        | InferenceEngineError::ResidentForkRequired { .. }
        | InferenceEngineError::ResidentStreamingReplacementRequired { .. }
        | InferenceEngineError::StreamingResidentPromotionRequired { .. }
        | InferenceEngineError::MlxMemoryLimitRejected { .. }
        | InferenceEngineError::EngineBusy => (
            PerformanceAttributionOutcome::Rejected,
            rejection_description,
        ),
        InferenceEngineError::Fatal { .. } => {
            (PerformanceAttributionOutcome::Failed, failure_description)
        }
    }
}

/// One generation's attribution, as finished by the engine.
pub struct GenerationAttributionRecord {
    pub performance_attribution: PerformanceAttribution,
    pub outcome: PerformanceAttributionOutcome,
    pub request_id: ipc_protocol::RequestId,
    pub configured_maximum_output_tokens: u16,
    pub mlx_memory_snapshot: Option<MlxMemorySnapshot>,
    pub failure_description: Option<&'static str>,
}

/// Appends one generation's attribution to the log, when attribution is on.
pub fn record_generation_performance_attribution<Engine: TextFinalizationHooks>(
    engine: &mut Engine,
    attribution_record: GenerationAttributionRecord,
) {
    if !attribution_record.performance_attribution.is_enabled() {
        return;
    }
    let request_id = attribution_record.request_id;
    let execution_core = engine.execution_core();
    let (Some(model_id), Some(model_revision)) = (
        execution_core.model_id.clone(),
        execution_core.model_revision.clone(),
    ) else {
        tracing::error!(
            request_id = request_id.value(),
            "completed generation lacked loaded-model attribution metadata"
        );
        return;
    };
    let measured_prompt_mlx_working_buffer_bytes =
        engine.measured_prompt_mlx_working_buffer_bytes();
    let mlx_memory_snapshot = attribution_record.mlx_memory_snapshot.as_ref();
    let Some(performance_attribution_report) = attribution_record
        .performance_attribution
        .finish_generation(GenerationPerformanceAttributionMetadata {
            outcome: attribution_record.outcome,
            model_id,
            model_revision,
            prefill_transient_observation_completed: measured_prompt_mlx_working_buffer_bytes
                .is_some(),
            prefill_observed_transient_high_water_bytes: measured_prompt_mlx_working_buffer_bytes
                .unwrap_or(0),
            request_id: request_id.value(),
            configured_maximum_output_tokens: attribution_record.configured_maximum_output_tokens,
            mlx_active_memory_bytes: mlx_memory_snapshot
                .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
            mlx_buffer_cache_bytes: mlx_memory_snapshot
                .and_then(|snapshot| u64::try_from(snapshot.mlx_buffer_cache_bytes()).ok()),
            mlx_peak_memory_bytes: mlx_memory_snapshot
                .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
            failure_description: attribution_record.failure_description.map(str::to_owned),
        })
    else {
        return;
    };
    if let Err(performance_attribution_write_error) = engine
        .execution_core_mut()
        .performance_attribution_log
        .record(&performance_attribution_report)
    {
        tracing::warn!(
            error = %performance_attribution_write_error,
            "failed to append generation performance attribution"
        );
    }
}

#[cfg(test)]
#[path = "generation_finalization_steps_tests.rs"]
mod generation_finalization_steps_tests;
