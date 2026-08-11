//! Best-effort diagnostic snapshot for request-scoped drafter failures.
//!
//! MLX does not expose physical allocation ownership tags. The breakdown here is
//! therefore a reconciliation of known logical owners against one process-wide
//! active-memory snapshot, not a claim that each category was measured alone.

use super::super::Qwen3_5EngineState;
use super::super::engine_request::Qwen3_5EngineRequest;

impl Qwen3_5EngineState {
    /// Emits one complete request and MLX snapshot after drafter work fails.
    ///
    /// Diagnostics are best effort so telemetry collection can never replace the
    /// original fail-closed SpecPrefill error.
    pub(in crate::qwen3_5) fn log_speculative_prefill_drafter_failure_diagnostics(
        &self,
        active_request: &Qwen3_5EngineRequest,
    ) {
        // Snapshot through the target runtime because target and request-scoped
        // drafter share process-global MLX accounting. Every conversion saturates
        // or falls back so diagnostics cannot panic while handling another error.
        let current_mlx_memory_snapshot_attempt = self.model.as_ref().map(|target_model| {
            target_model
                .runtime()
                .memory_snapshot()
                .map(|mlx_memory_snapshot| {
                    let active_memory_bytes =
                        u64::try_from(mlx_memory_snapshot.active_memory_bytes())
                            .unwrap_or(u64::MAX);
                    let active_memory_breakdown = target_model.active_memory_breakdown(
                        active_request.request_decoder_state(),
                        active_request.additional_context_state_payload_bytes(),
                        active_memory_bytes,
                        0,
                    );
                    (
                        active_memory_bytes,
                        u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes())
                            .unwrap_or(u64::MAX),
                        u64::try_from(mlx_memory_snapshot.peak_memory_bytes()).unwrap_or(u64::MAX),
                        active_memory_breakdown,
                    )
                })
        });
        let mlx_memory_snapshot_available = current_mlx_memory_snapshot_attempt
            .as_ref()
            .is_some_and(Result::is_ok);
        let mlx_memory_snapshot_error = current_mlx_memory_snapshot_attempt
            .as_ref()
            .and_then(|mlx_memory_snapshot_attempt| mlx_memory_snapshot_attempt.as_ref().err())
            .map(ToString::to_string);
        let current_mlx_memory = current_mlx_memory_snapshot_attempt
            .and_then(Result::ok)
            .unwrap_or_default();
        let current_active_memory_breakdown = current_mlx_memory.3;
        // Known target categories are derived from model/request geometry. The
        // remainder includes runtime graph work and any still-live drafter work;
        // it must remain explicitly unclassified rather than falsely attributed.
        let current_unclassified_runtime_work_bytes = current_mlx_memory
            .0
            .saturating_sub(current_active_memory_breakdown.expert_payload_bytes)
            .saturating_sub(current_active_memory_breakdown.model_core_payload_bytes)
            .saturating_sub(current_active_memory_breakdown.context_state_payload_bytes);

        // The request captures a standalone drafter-phase snapshot before draft
        // owners are dropped. Absence means capture itself was unavailable, not
        // that the drafter necessarily consumed zero bytes.
        let draft_memory_telemetry = active_request.speculative_prefill_draft_memory_telemetry;
        let draft_memory_snapshot_available = draft_memory_telemetry.is_some();
        let draft_memory_telemetry = draft_memory_telemetry.unwrap_or(
            crate::MlxMemoryTelemetry::new(0, 0, 0, crate::MlxActiveMemoryBreakdown::default()),
        );
        let draft_active_memory_breakdown = draft_memory_telemetry.active_memory_breakdown;

        // Expert-cache statistics provide a relational before/after view. They
        // are reliable for retained payload changes even though process-wide MLX
        // active memory can include unrelated graph allocations.
        let target_expert_memory_statistics = self
            .model
            .as_ref()
            .map(|target_model| target_model.expert_weight_memory_cache_statistics())
            .unwrap_or_default();
        let draft_persistent_prompt_cache_available = self
            .speculative_prefill_draft_persistent_prompt_cache
            .is_some();
        // Cache counts describe durable logical content and are safe to collect
        // without opening a new store or changing request state.
        let draft_persistent_prompt_cache_sequence_state_count = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
            .map_or(0, |draft_persistent_prompt_cache| {
                draft_persistent_prompt_cache.sequence_state_block_count()
            });
        let draft_persistent_prompt_cache_boundary_snapshot_count = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
            .map_or(0, |draft_persistent_prompt_cache| {
                draft_persistent_prompt_cache.boundary_state_snapshot_count()
            });
        let draft_persistent_prompt_cache_size_bytes = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
            .map_or(0, |draft_persistent_prompt_cache| {
                draft_persistent_prompt_cache.total_size_bytes()
            });
        let restored_target_position_count = active_request
            .speculative_prefill_restored_target_token_positions
            .as_ref()
            .map_or(0, |restored_target_token_positions| {
                restored_target_token_positions.shape()[0].max(0) as usize
            });
        // Scored suffix is derived from work-reuse counters so diagnostics remain
        // meaningful even if scoring failed before a selection vector existed.
        let selected_target_position_count = active_request
            .speculative_prefill_selected_token_positions
            .as_ref()
            .map_or(0, Vec::len);
        let drafter_scored_suffix_token_count = active_request
            .prompt_work_reuse
            .drafter_eligible_token_count
            .saturating_sub(
                active_request
                    .prompt_work_reuse
                    .drafter_restored_token_count,
            );
        let target_expert_payload_bytes_at_request_start = active_request
            .expert_weight_memory_cache_statistics_at_request_start
            .resident_payload_byte_count;

        // Emit one event rather than a sequence of partial events. Correlating
        // all fields by request ID lets a later investigation reconstruct the
        // exact lifecycle point without changing the error returned to the user.
        tracing::error!(
            request_id = active_request.request_id.value(),
            target_model_id = self.model_id.as_deref().unwrap_or("unavailable"),
            draft_model_id = self
                .speculative_prefill
                .draft_model_id
                .as_deref()
                .unwrap_or("unavailable"),
            draft_model_revision = self
                .speculative_prefill_draft_model_revision
                .as_deref()
                .unwrap_or("unavailable"),
            prompt_token_count = active_request.input_token_ids.len(),
            prefill_cursor_token_count = active_request.prefill_cursor,
            ordinary_target_control_span_token_count = active_request
                .ordinary_target_prefill_control_span_token_count,
            target_eligible_token_count = active_request.prompt_work_reuse.target_eligible_token_count,
            target_restored_token_count = active_request.prompt_work_reuse.target_restored_token_count,
            restored_target_position_count,
            drafter_eligible_token_count = active_request
                .prompt_work_reuse
                .drafter_eligible_token_count,
            drafter_restored_token_count = active_request
                .prompt_work_reuse
                .drafter_restored_token_count,
            drafter_scored_suffix_token_count,
            selected_target_position_count,
            processed_visual_image_count = active_request
                .speculative_prefill_processed_visual_images
                .len(),
            mlx_active_memory_limit_bytes = self.memory_limits.active_memory_limit_bytes(),
            mlx_allowed_active_memory_bytes = self.memory_limits.allowed_active_memory_bytes(),
            mlx_memory_snapshot_available,
            mlx_memory_snapshot_error = ?mlx_memory_snapshot_error,
            current_mlx_active_memory_bytes = current_mlx_memory.0,
            current_mlx_allocator_cache_memory_bytes = current_mlx_memory.1,
            current_mlx_peak_memory_bytes = current_mlx_memory.2,
            current_target_expert_payload_bytes = current_active_memory_breakdown.expert_payload_bytes,
            current_target_model_core_payload_bytes = current_active_memory_breakdown.model_core_payload_bytes,
            current_target_context_state_payload_bytes = current_active_memory_breakdown.context_state_payload_bytes,
            current_unclassified_runtime_work_bytes,
            target_request_decoder_state_payload_bytes = active_request
                .request_decoder_state()
                .payload_byte_count(),
            target_additional_context_state_payload_bytes = active_request
                .additional_context_state_payload_bytes(),
            draft_memory_snapshot_available,
            draft_phase_mlx_active_memory_bytes = draft_memory_telemetry.active_memory_bytes,
            draft_phase_mlx_allocator_cache_memory_bytes = draft_memory_telemetry.allocator_cache_memory_bytes,
            draft_phase_mlx_peak_memory_bytes = draft_memory_telemetry.peak_memory_bytes,
            draft_phase_target_expert_payload_bytes = draft_active_memory_breakdown.expert_payload_bytes,
            draft_phase_target_model_core_payload_bytes = draft_active_memory_breakdown.model_core_payload_bytes,
            draft_phase_target_context_state_payload_bytes = draft_active_memory_breakdown.context_state_payload_bytes,
            draft_phase_drafter_memory_bytes = draft_active_memory_breakdown.speculative_prefill_draft_memory_bytes,
            target_expert_entry_count = target_expert_memory_statistics.entry_count,
            target_expert_payload_bytes_at_request_start,
            target_expert_resident_payload_bytes = target_expert_memory_statistics.resident_payload_byte_count,
            target_expert_reclaimed_payload_bytes = target_expert_payload_bytes_at_request_start
                .saturating_sub(target_expert_memory_statistics.resident_payload_byte_count),
            target_expert_eviction_count = target_expert_memory_statistics.eviction_count,
            draft_persistent_prompt_cache_available,
            draft_persistent_prompt_cache_sequence_state_count,
            draft_persistent_prompt_cache_boundary_snapshot_count,
            draft_persistent_prompt_cache_size_bytes,
            "captured configured SpecPrefill drafter failure diagnostics"
        );
    }
}
