//! One terminal cleanup path for success, rejection, cancellation, and failure.
//!
//! Request-owned lazy arrays are dropped before the memory governor ends the
//! request, dense layers are kept again, and allocator cleanup runs. Final
//! telemetry then reports the retained pager ownership after cleanup.

use crate::{GenerationFinalization, PerformanceAttributionOutcome};
use ipc_protocol::{ReplyMemoryRecord, RequestId};
use text_generation::generation_loop::{generation_finalization_steps, request_end};

use super::Qwen3_5EngineState;
use super::engine_request::StreamingEngineRequest;

impl generation_finalization_steps::TextFinalizationHooks for Qwen3_5EngineState {
    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        self.core.model.as_ref().and_then(
            crate::qwen3_5_streaming::model::Qwen3_5Model::measured_prompt_mlx_working_buffer_bytes,
        )
    }

    fn finalize_generation_request(
        &mut self,
        active_request: StreamingEngineRequest,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        request_end::finalize_generation_request(
            self,
            active_request,
            performance_attribution_outcome,
            failure_description,
        )
    }

    /// Keeps the session's expert usage for the next load, and reports how
    /// the reply's experts were streamed.
    fn complete_generation_finalization(
        &self,
        generation_finalization: GenerationFinalization,
        expert_memory_mode: Option<ipc_protocol::ExpertMemoryMode>,
        expert_residency_telemetry: Option<
            crate::inference_engine::contract::ExpertResidencyTelemetry,
        >,
    ) -> GenerationFinalization {
        self.persist_expert_usage_history();
        let Some(model) = self.core.model.as_ref() else {
            return generation_finalization;
        };
        model.attach_expert_streaming_stats(
            generation_finalization,
            self.core.model_id.as_deref(),
            expert_memory_mode,
            expert_residency_telemetry,
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

/// One line per reply with where its time went inside the engine: the
/// forwards it ran, the in-graph lookups' reads and bookkeeping, a drafted
/// round's stages, and the hand-offs MLX made. Always on, like the line
/// above, so a slow reply names its own stage before anyone adds a timer.
fn log_stages_over_the_reply(
    request_id: u64,
    stages: &crate::qwen3_5_streaming::model::reply_stage_tally::ReplyStageCounts,
) {
    if stages.reply_forward_count == 0 {
        return;
    }
    let forward_count = usize::try_from(stages.reply_forward_count).unwrap_or(usize::MAX);
    let hand_offs = stages.hand_offs;
    tracing::info!(
        request_id,
        reply_forward_count = stages.reply_forward_count,
        reply_row_count = stages.reply_row_count,
        lookups_per_forward =
            format_args!("{:.1}", per_reply_token(forward_count, stages.lookup_count)),
        lookup_read_milliseconds_per_forward = format_args!(
            "{:.3}",
            per_reply_token(forward_count, stages.lookup_storage_read_nanoseconds) / 1_000_000.0
        ),
        lookup_bookkeeping_milliseconds_per_forward = format_args!(
            "{:.3}",
            per_reply_token(forward_count, stages.lookup_bookkeeping_nanoseconds()) / 1_000_000.0
        ),
        draft_round_count = stages.draft_round_count,
        draft_milliseconds = stages.draft_nanoseconds / 1_000_000,
        verify_milliseconds = stages.verify_nanoseconds / 1_000_000,
        pick_milliseconds = stages.pick_nanoseconds / 1_000_000,
        keep_milliseconds = stages.keep_nanoseconds / 1_000_000,
        blocking_evaluations_per_forward = format_args!(
            "{:.2}",
            per_reply_token(forward_count, hand_offs.blocking_evaluations)
        ),
        asynchronous_evaluations_per_forward = format_args!(
            "{:.2}",
            per_reply_token(forward_count, hand_offs.asynchronous_evaluations)
        ),
        cross_stream_waits_per_forward = format_args!(
            "{:.1}",
            per_reply_token(forward_count, hand_offs.cross_stream_waits)
        ),
        "the reply's stages"
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

impl request_end::TextRequestEndHooks for Qwen3_5EngineState {
    type ReplyRecord = ReplyMemoryRecord;

    fn take_reply_record(
        &mut self,
        active_request: &mut StreamingEngineRequest,
    ) -> ReplyMemoryRecord {
        let request_id = active_request.request_id;
        let mlx_buffer_cache_reply_record = active_request
            .engine_extension
            .mlx_buffer_cache_reply_record;
        log_mlx_buffer_cache_over_the_reply(request_id.value(), mlx_buffer_cache_reply_record);
        let reply_memory_record = mlx_buffer_cache_reply_record.reply_memory_record(
            std::mem::take(&mut active_request.engine_extension.reply_expert_route_record),
        );
        log_astro_expert_cache_over_the_reply(request_id.value(), &reply_memory_record);
        if let Some(model) = self.core.model.as_ref()
            && let Some(reply_stages) = model.reply_measurement.stage_tally.end_reply()
        {
            log_stages_over_the_reply(request_id.value(), &reply_stages);
        }
        reply_memory_record
    }

    /// Request-owned lazy arrays are gone by now. Hot-expert warmed tables
    /// stay: the routed-expert distribution is model-and-domain knowledge,
    /// not request state, so an identical or similar next ask serves from
    /// RAM instead of re-reading its experts (issue #955). The next
    /// request's start gives the astro expert cache its capacity for that
    /// prompt, and the astro expert cache gives back only what no longer
    /// fits.
    fn end_governed_request_before_memory_release(&mut self, request_id: RequestId) {
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
    }

    fn attach_reply_record(
        generation_finalization: GenerationFinalization,
        reply_memory_record: ReplyMemoryRecord,
    ) -> GenerationFinalization {
        generation_finalization.with_reply_memory_record(reply_memory_record)
    }
}
