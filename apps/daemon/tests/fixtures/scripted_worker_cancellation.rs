//! Telemetry the scripted worker can emit between a `Cancel` command and
//! its acknowledgement. The daemon classifies every event in that window
//! through its cancellation drain, so these helpers exist to prove benign
//! telemetry (persistent prompt-cache write-out, the per-generation
//! expert-streaming report, MLX memory samples) gets recorded rather than
//! treated as a protocol violation that replaces the worker.

use std::error::Error;

use astronomical_ipc_protocol::{
    ExpertLayerSeatingCounts, ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolWriter,
    WorkerEvent, WorkerExpertResidencySnapshot, WorkerExpertStreamingStats,
    WorkerMlxMemorySnapshot,
};

/// Fixture that publishes expert-streaming stats during the cancellation
/// drain. It shares a dispatch arm with the prompt-cache fixture, so the
/// ids stay together where that arm lives.
pub(super) const EXPERT_STREAMING_TELEMETRY_FIXTURE_MODEL_ID: &str =
    "astronomical/expert-streaming-stats-during-cancellation-fixture";

/// Selects the memory telemetry emitted before a cancellation acknowledgement.
pub(super) enum CancellationMlxTelemetry {
    None,
    Publish,
    Clear,
}

/// Emits the one-shot telemetry the cancellation fixtures scheduled before
/// the acknowledgement. Each flag clears as it fires so a repeated Cancel
/// cannot replay events into a later drain.
pub(super) async fn emit_cancellation_telemetry(
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
    should_emit_prompt_cache_stats: &mut bool,
    should_emit_expert_streaming_stats: &mut bool,
    cancellation_mlx_telemetry: &mut CancellationMlxTelemetry,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if *should_emit_prompt_cache_stats {
        *should_emit_prompt_cache_stats = false;
        send_prompt_cache_stats(event_writer).await?;
    }
    if *should_emit_expert_streaming_stats {
        *should_emit_expert_streaming_stats = false;
        event_writer
            .send_event(&WorkerEvent::ExpertStreamingStats {
                stats: expert_streaming_stats_snapshot(),
            })
            .await?;
    }
    send_mlx_telemetry(event_writer, cancellation_mlx_telemetry).await?;
    Ok(())
}

/// The cache publication a real worker flushes when a generation ends,
/// landing after the client already dropped the stream.
async fn send_prompt_cache_stats(
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    event_writer
        .send_event(&WorkerEvent::PersistentPromptCacheStats {
            persistent_prompt_cache_hits: 1,
            persistent_prompt_cache_misses: 0,
            persistent_prompt_cache_tokens_saved: 2_048,
            persistent_prompt_cache_block_token_count: 2_048,
            persistent_prompt_cache_sequence_state_block_count: 1,
            persistent_prompt_cache_boundary_state_snapshot_count: 1,
            persistent_prompt_cache_visual_embedding_count: 0,
            persistent_prompt_cache_total_size_bytes: 4_096,
            persistent_prompt_cache_visual_embedding_total_size_bytes: 0,
            persistent_prompt_cache_maximum_size_bytes: 50_000_000_000,
            persistent_prompt_cache_visual_embedding_hits: 0,
            persistent_prompt_cache_visual_embedding_misses: 0,
            persistent_prompt_cache_visual_embedding_rows_loaded: 0,
            persistent_prompt_cache_partial_tail_hits: 0,
        })
        .await?;
    Ok(())
}

async fn send_mlx_telemetry(
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
    cancellation_mlx_telemetry: &mut CancellationMlxTelemetry,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match std::mem::replace(cancellation_mlx_telemetry, CancellationMlxTelemetry::None) {
        CancellationMlxTelemetry::None => {}
        CancellationMlxTelemetry::Publish => {
            event_writer
                .send_event(&WorkerEvent::MlxMemorySample {
                    mlx_memory_snapshot: Some(cancellation_memory_snapshot(44_000)),
                    expert_residency: None,
                })
                .await?;
        }
        CancellationMlxTelemetry::Clear => {
            event_writer
                .send_event(&WorkerEvent::MlxMemorySample {
                    mlx_memory_snapshot: None,
                    expert_residency: None,
                })
                .await?;
        }
    }
    Ok(())
}

/// The per-generation expert-streaming report a real worker pushes exactly
/// once at finalization; the single nonzero routing counter is the value
/// the integration journey asserts survived the drain.
fn expert_streaming_stats_snapshot() -> WorkerExpertStreamingStats {
    WorkerExpertStreamingStats {
        model_id: None,
        expert_memory_mode: ExpertMemoryMode::Hybrid,
        expert_residency: WorkerExpertResidencySnapshot {
            total_layer_count: 8,
            resident_expert_count: 16,
            resident_expert_payload_bytes: 1_024,
            complete_expert_payload_byte_count: 2_048,
            expert_capacity: 64,
            layer_seating: ExpertLayerSeatingCounts::default(),
        },
        layer_topology: Vec::new(),
        retained_route_assignment_count: 9,
        storage_route_assignment_count: 1,
        hot_route_fully_covered_layer_token_count: 0,
        hot_route_partially_covered_layer_token_count: 0,
        hot_route_missed_layer_token_count: 0,
        expert_source_read_call_count: 0,
        expert_source_read_byte_count: 0,
        expert_source_read_elapsed_nanoseconds: 0,
        prefill_source_read_elapsed_nanoseconds: 0,
        decode_source_read_elapsed_nanoseconds: 0,
        expert_source_physical_disk_read_bytes: None,
        avoided_complete_layer_source_bytes: 0,
        mandatory_prefill_source_bytes: 0,
        mandatory_decode_source_bytes: 0,
        prompt_token_count: 1,
        generated_token_count: 1,
    }
}

/// Builds deterministic memory telemetry for cancellation-containment tests.
pub(super) fn cancellation_memory_snapshot(active_memory_bytes: u64) -> WorkerMlxMemorySnapshot {
    WorkerMlxMemorySnapshot {
        source: MlxMemorySnapshotSource::Finalized,
        active_memory_bytes,
        allocator_cache_memory_bytes: 2_000,
        peak_memory_bytes: active_memory_bytes + 1_000,
        expert_payload_bytes: 20_000,
        model_core_payload_bytes: 10_000,
        context_state_payload_bytes: 5_000,
        memory_ceiling_utilization: None,
    }
}
