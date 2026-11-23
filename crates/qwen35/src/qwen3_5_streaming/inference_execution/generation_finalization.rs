//! One terminal cleanup path for success, rejection, cancellation, and failure.
//!
//! Request-owned lazy arrays are dropped before the memory governor ends the
//! request, dense layers are kept again, and allocator cleanup runs. Final
//! telemetry then reports the retained pager ownership after cleanup.

use super::NO_RETAINED_LIVE_PREFIX_BYTES;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::qwen3_5_core::shared_execution::generation_finalization_steps;
use crate::{
    GenerationFinalization, MlxMemoryTelemetry, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceOperation,
};

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::shared_execution::request_memory_release::release_request_memory;

impl generation_finalization_steps::Qwen3_5FinalizationHooks for Qwen3_5EngineState {
    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        self.core.model.as_ref().and_then(
            crate::qwen3_5_streaming::model::Qwen3_5Model::measured_prompt_mlx_working_buffer_bytes,
        )
    }

    fn finalize_generation_request(
        &mut self,
        active_request: Qwen3_5EngineRequest,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        Qwen3_5EngineState::finalize_generation_request(
            self,
            active_request,
            performance_attribution_outcome,
            failure_description,
        )
    }
}

impl Qwen3_5EngineState {
    /// Closes the request with the memory governor: its stored state is
    /// gone, and the astro expert cache may keep what the idle model has room for.
    fn govern_request_end_at_finalization(&self, request_id: u64) {
        if let Some(model) = self.core.model.as_ref() {
            // A request that ends while a prompt is being read one layer at
            // a time leaves that prompt's pages in hand and routes pending.
            model.discard_expert_state_of_an_unfinished_prompt();
            model.govern_request_end(request_id);
        }
    }

    /// Reads back the dense layers this request gave up for its context, now
    /// that its state is gone and the memory governor is idle again (issue
    /// #1985). Without this one long prompt left every later request
    /// streaming more layers than there is room to keep. It costs at most a
    /// few layer reads per request, while decode already reads every
    /// streamed layer for every token. A failure leaves the layers
    /// streaming, which is still correct, so it is only logged.
    fn keep_dense_layers_again_at_finalization(&self, request_id: u64) {
        let Some(model) = self.core.model.as_ref() else {
            return;
        };
        if let Err(keep_error) = model.keep_dense_layers_the_governor_has_room_for() {
            tracing::warn!(
                request_id,
                error = %keep_error,
                "could not keep the dense layers again after the request"
            );
        }
    }

    pub(super) fn finalize_generation_request(
        &mut self,
        mut active_request: Qwen3_5EngineRequest,
        outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        let request_id = active_request.request_id;
        let configured_maximum_output_tokens = active_request.maximum_output_tokens;
        // Attribution must outlive the request because cleanup and the final
        // memory snapshot are part of the user-visible request cost.
        let mut performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let mlx_buffer_cache_reply_record = active_request
            .engine_extension
            .mlx_buffer_cache_reply_record;
        log_mlx_buffer_cache_over_the_reply(request_id.value(), mlx_buffer_cache_reply_record);
        let reply_memory_record = mlx_buffer_cache_reply_record.reply_memory_record(
            std::mem::take(&mut active_request.engine_extension.reply_expert_route_record),
        );
        log_astro_expert_cache_over_the_reply(request_id.value(), &reply_memory_record);
        // Drop pending tokens, decoder state, and all
        // request-local snapshot references before lifting the pressure ceiling.
        // Otherwise newly admitted pages could compete with memory that is only
        // logically dead but still owned by this request.
        drop(active_request);
        // Request-owned lazy arrays are gone. Hot-expert warmed tables stay:
        // the routed-expert distribution is model-and-domain knowledge, not
        // request state, so an identical or similar next ask serves from RAM
        // instead of re-reading its experts (issue #955). The next request's
        // start gives the astro expert cache its capacity for that prompt, and the astro expert cache
        // gives back only what no longer fits.
        self.govern_request_end_at_finalization(request_id.value());
        self.keep_dense_layers_again_at_finalization(request_id.value());
        if let Some(model) = self.core.model.as_ref() {
            let astro_expert_cache_statistics = model.astro_expert_cache_statistics();
            tracing::info!(
                request_id = request_id.value(),
                expert_memory_mode = ?model.expert_memory_mode(),
                retained_expert_payload_bytes =
                    astro_expert_cache_statistics.resident_payload_byte_count,
                maximum_retained_expert_payload_bytes =
                    astro_expert_cache_statistics.maximum_resident_payload_byte_count,
                "kept the astro expert cache for the next request"
            );
        }
        // `release_request_memory` synchronizes the model stream before clearing
        // allocator storage, so one-token-ahead decode cannot race reclamation.
        let mlx_memory_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::MlxBufferCacheCleanup,
            |_performance_attribution| {
                release_request_memory(self.core.model.as_deref(), request_id)
            },
        );
        let generation_finalization = self.collect_generation_finalization(
            &mut performance_attribution,
            mlx_memory_snapshot
                .as_ref()
                .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
        );
        let generation_finalization =
            generation_finalization.with_reply_memory_record(reply_memory_record);
        generation_finalization_steps::record_generation_performance_attribution(
            self,
            generation_finalization_steps::GenerationAttributionRecord {
                performance_attribution,
                outcome,
                request_id,
                configured_maximum_output_tokens,
                mlx_memory_snapshot,
                failure_description,
            },
        );
        generation_finalization
    }

    fn collect_generation_finalization(
        &self,
        performance_attribution: &mut PerformanceAttribution,
        release_peak_memory_bytes: Option<u64>,
    ) -> GenerationFinalization {
        let Some(model) = self.core.model.as_ref() else {
            return GenerationFinalization::default();
        };
        let expert_memory_mode = Some(model.observed_expert_memory_mode());
        let mlx_memory_telemetry = match performance_attribution.measure_operation(
            PerformanceOperation::FinalizedMlxMemorySnapshot,
            |_performance_attribution| model.memory_snapshot_with_forward_peak(),
        ) {
            Ok(mlx_memory_snapshot) => match (
                u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes()),
                u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
            ) {
                (
                    Ok(active_memory_bytes),
                    Ok(mlx_buffer_cache_bytes),
                    Ok(sampled_peak_memory_bytes),
                ) => {
                    // The MLX peak counter resets at `mlx_buffer_cache` cleanups,
                    // so this post-cleanup sample can read a lower window peak
                    // than the synchronized release sample taken moments
                    // earlier. The finalized claim is a request maximum, so it
                    // must retain the higher of the two windows.
                    let peak_memory_bytes = release_peak_memory_bytes.map_or(
                        sampled_peak_memory_bytes,
                        |release_peak_memory_bytes| {
                            release_peak_memory_bytes.max(sampled_peak_memory_bytes)
                        },
                    );
                    // The finalized claim derives from the breakdown reconciling
                    // this exact snapshot, so the pair cannot straddle a
                    // promotion or demote (issue #337).
                    let active_memory_breakdown = model.finalized_active_memory_breakdown(
                        active_memory_bytes,
                        NO_RETAINED_LIVE_PREFIX_BYTES,
                        0,
                    );
                    let expert_residency_telemetry = Some(
                        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
                    );
                    (
                        Some(
                            MlxMemoryTelemetry::new(
                                active_memory_bytes,
                                mlx_buffer_cache_bytes,
                                peak_memory_bytes,
                                active_memory_breakdown,
                            )
                            .with_memory_ceiling_utilization(
                                model.between_request_memory_ceiling_utilization_for_breakdown(
                                    active_memory_bytes,
                                    active_memory_breakdown,
                                ),
                            )
                            .with_memory_ledger_reading(model.latest_memory_ledger_reading()),
                        ),
                        expert_residency_telemetry,
                    )
                }
                memory_counter_results => {
                    tracing::warn!(
                        ?memory_counter_results,
                        "could not publish finalized MLX memory because a runtime counter exceeds the u64 range"
                    );
                    (None, None)
                }
            },
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "could not capture finalized MLX memory after expert residency recovery"
                );
                (None, None)
            }
        };
        let (mlx_memory_telemetry, expert_residency_telemetry) = mlx_memory_telemetry;
        self.persist_expert_usage_history();
        model.attach_expert_streaming_stats(
            GenerationFinalization::new(
                expert_memory_mode,
                mlx_memory_telemetry,
                expert_residency_telemetry,
            ),
            self.core.model_id.as_deref(),
            expert_memory_mode,
            expert_residency_telemetry,
        )
    }

    /// Persists the phase-split expert demand records beside the artifact so
    /// the next session's warm tables start hot instead of at the
    /// first-touch floor (colibri's `.coli_usage` analog). The snapshot ages
    /// both records in place — prior evidence halves, the current generation
    /// is added — so heat must be re-earned to survive. Best-effort by
    /// contract: the store logs write failures and serving continues
    /// unchanged.
    fn persist_expert_usage_history(&self) {
        let Some(model) = self.core.model.as_ref() else {
            return;
        };
        let Some(astro_expert_cache) = model.astro_expert_cache.as_ref() else {
            return;
        };
        let usage_file_path = crate::qwen3_5_streaming::expert_streaming::expert_usage_file_path(
            &self.core.model_directory,
        );
        let persisted_usage = astro_expert_cache
            .borrow_mut()
            .take_persisted_expert_usage();
        crate::qwen3_5_streaming::expert_streaming::write_expert_usage(
            &usage_file_path,
            self.core.model_id.as_deref(),
            &persisted_usage,
        );
    }
}

/// One line per reply with how well the `astro_expert_cache` chose what to
/// keep: per reply token, the routed experts it held, the ones it missed, the
/// bytes those misses read and the time spent waiting on them. Always on, so a
/// person's own logs show the figure without performance attribution.
fn log_astro_expert_cache_over_the_reply(
    request_id: u64,
    reply_memory_record: &ipc_protocol::ReplyMemoryRecord,
) {
    let reply_token_count = reply_memory_record
        .storage_read_expert_counts_by_token
        .len();
    if reply_token_count == 0 || reply_memory_record.routed_expert_count_per_token == 0 {
        return;
    }
    tracing::info!(
        request_id,
        reply_token_count,
        expert_hits_per_token = format_args!(
            "{:.2}",
            per_reply_token(reply_token_count, reply_memory_record.expert_hit_count)
        ),
        expert_misses_per_token = format_args!(
            "{:.2}",
            per_reply_token(reply_token_count, reply_memory_record.expert_miss_count)
        ),
        storage_read_megabytes_per_token = format_args!(
            "{:.2}",
            per_reply_token(reply_token_count, reply_memory_record.storage_read_bytes)
                / 1_048_576.0
        ),
        storage_read_wait_milliseconds_per_token = format_args!(
            "{:.2}",
            per_reply_token(
                reply_token_count,
                reply_memory_record.storage_read_wait_nanoseconds
            ) / 1_000_000.0
        ),
        "the astro_expert_cache over the reply"
    );
}

/// `total` shared out over the reply's tokens.
fn per_reply_token(reply_token_count: usize, total: u64) -> f64 {
    total as f64 / reply_token_count as f64
}

/// Says how full the `mlx_buffer_cache` was after each token of the reply.
/// Tokens after which it was empty allocated every buffer afresh, which is
/// what made replies at the top of the memory range a third as fast.
fn log_mlx_buffer_cache_over_the_reply(
    request_id: u64,
    mlx_buffer_cache_reply_record: crate::memory::mlx_buffer_cache_reply_record::MlxBufferCacheReplyRecord,
) {
    if mlx_buffer_cache_reply_record.sampled_token_count() == 0 {
        return;
    }
    tracing::info!(
        request_id,
        sampled_token_count = mlx_buffer_cache_reply_record.sampled_token_count(),
        empty_token_count = mlx_buffer_cache_reply_record.empty_token_count(),
        smallest_mlx_buffer_cache_bytes = mlx_buffer_cache_reply_record.smallest_bytes(),
        mean_mlx_buffer_cache_bytes = mlx_buffer_cache_reply_record.mean_bytes(),
        largest_mlx_buffer_cache_bytes = mlx_buffer_cache_reply_record.largest_bytes(),
        "the MLX buffer cache over the reply"
    );
}
