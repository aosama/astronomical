//! How any text engine ends a request (issue #2473): keep what the
//! engine reports about the reply, drop the request, release its MLX memory
//! under measurement, collect the finalization, and record the request's
//! attribution.
//!
//! Engines differ in when their governed work ends. The Qwen3.5 text engine
//! ends it and keeps its dense layers again before the memory is released.
//! Each engine says which through the two `end_governed_request_*` hooks.

use ipc_protocol::RequestId;

use crate::engine_request::TextEngineRequest;
use crate::{
    GenerationFinalization, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation,
};

use super::generation_finalization_collection::collect_generation_finalization;
use super::generation_finalization_steps::{
    GenerationAttributionRecord, TextFinalizationHooks, record_generation_performance_attribution,
};
use super::request_memory_release::release_request_memory;

/// The points where the engines' request ends differ.
pub trait TextRequestEndHooks: TextFinalizationHooks {
    /// What the engine reports about the reply beyond the shared figures.
    type ReplyRecord;

    /// Takes the reply record from the request before it is dropped.
    fn take_reply_record(
        &mut self,
        active_request: &mut TextEngineRequest<Self::RequestExtension>,
    ) -> Self::ReplyRecord;

    /// Ends governed work once the request is dropped, before its memory is
    /// released.
    fn end_governed_request_before_memory_release(&mut self, _request_id: RequestId) {}

    /// Ends governed work after the request's memory is released.
    fn end_governed_request_after_memory_release(&mut self) {}

    /// Adds the reply record to the finalization report.
    fn attach_reply_record(
        generation_finalization: GenerationFinalization,
        reply_record: Self::ReplyRecord,
    ) -> GenerationFinalization;
}

/// Ends the request in the engine's order and reports its finalization.
pub fn finalize_generation_request<Engine: TextRequestEndHooks>(
    engine: &mut Engine,
    mut active_request: TextEngineRequest<Engine::RequestExtension>,
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
    let reply_record = engine.take_reply_record(&mut active_request);
    // Drop pending tokens and decoder state before sampling post-request
    // memory, so memory that is only logically dead is not counted as held.
    drop(active_request);
    engine.end_governed_request_before_memory_release(request_id);
    // `release_request_memory` synchronizes the model stream before clearing
    // allocator storage, so one-token-ahead decode cannot race reclamation.
    let mlx_memory_snapshot = performance_attribution.measure_operation(
        PerformanceOperation::MlxBufferCacheCleanup,
        |_performance_attribution| {
            release_request_memory(engine.execution_core().model.as_ref(), request_id)
        },
    );
    engine.end_governed_request_after_memory_release();
    let generation_finalization = collect_generation_finalization(
        engine,
        &mut performance_attribution,
        mlx_memory_snapshot
            .as_ref()
            .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
    );
    let generation_finalization =
        Engine::attach_reply_record(generation_finalization, reply_record);
    record_generation_performance_attribution(
        engine,
        GenerationAttributionRecord {
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
