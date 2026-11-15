use super::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::support::{chat_command, close_worker_transport, next_event, ready_event};
use crate::EngineBackedWorker;
use astronomical_ipc_protocol::{
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, WorkerCommand, WorkerEvent,
};
use std::time::Duration;
use tokio::io::{duplex, split};
use tokio::time::timeout;
#[tokio::test]
async fn should_report_astro_ssd_prompt_cache_lookup_counters_when_generation_starts() {
    let astro_ssd_prompt_cache_stats_after_lookup = WorkerEvent::AstroSsdPromptCacheStats {
        astro_ssd_prompt_cache_hits: 1,
        astro_ssd_prompt_cache_misses: 1,
        astro_ssd_prompt_cache_tokens_saved: 2_048,
        astro_ssd_prompt_cache_partial_tail_hits: 1,
        astro_ssd_prompt_cache_block_token_count: 2_048,
        astro_ssd_prompt_cache_sequence_state_block_count: 2,
        astro_ssd_prompt_cache_boundary_state_snapshot_count: 1,
        astro_ssd_prompt_cache_visual_embedding_count: 0,
        astro_ssd_prompt_cache_total_size_bytes: 40_000,
        astro_ssd_prompt_cache_visual_embedding_total_size_bytes: 0,
        astro_ssd_prompt_cache_maximum_size_bytes: 50_000,
        astro_ssd_prompt_cache_visual_embedding_hits: 0,
        astro_ssd_prompt_cache_visual_embedding_misses: 0,
        astro_ssd_prompt_cache_visual_embedding_rows_loaded: 0,
    };
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(4_096),
        ScriptedChatEngine::with_cached_token_count(2_048)
            .with_active_generation_astro_ssd_prompt_cache_stats(
                astro_ssd_prompt_cache_stats_after_lookup.clone(),
            ),
    );
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
    let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    daemon_writer
        .send_command(&WorkerCommand::Generate(chat_command(91, 32)))
        .await
        .expect("the worker should receive a chat request");

    assert_eq!(
        next_event(&mut daemon_reader).await,
        astro_ssd_prompt_cache_stats_after_lookup,
        "the daemon must see the classified cache lookup before a long prefill completes"
    );
    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_delegate_a_loaded_model_cache_clear_to_the_engine_owner() {
    timeout(Duration::from_secs(2), async {
        let expected_clear_event = WorkerEvent::AstroSsdPromptCacheCleared {
            model_id: Some("example/scripted-chat".to_owned()),
            blocks_removed: 4,
            bytes_freed: 8_192,
        };
        let engine_worker = EngineBackedWorker::new(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new()
                .with_astro_ssd_prompt_cache_clear_event(expected_clear_event.clone()),
        );
        let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
        let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
        let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
        let mut daemon_reader = ProtocolReader::new(daemon_reader_transport);
        let mut daemon_writer = ProtocolWriter::new(daemon_writer_transport);
        let worker_task = tokio::spawn(async move {
            engine_worker
                .run(worker_reader_transport, worker_writer_transport)
                .await
        });

        assert_eq!(next_event(&mut daemon_reader).await, ready_event());
        daemon_writer
            .send_command(&WorkerCommand::ClearAstroSsdPromptCache {
                model_id: Some("example/scripted-chat".to_owned()),
            })
            .await
            .expect("worker should receive the cache-clear command");
        assert_eq!(next_event(&mut daemon_reader).await, expected_clear_event);

        close_worker_transport(daemon_writer, worker_task).await;
    })
    .await
    .expect("loaded-engine cache-clear journey should finish within two seconds");
}
