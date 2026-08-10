use super::*;

#[tokio::test]
async fn should_report_prompt_cache_lookup_counters_when_generation_starts() {
    let prompt_cache_stats_after_lookup = WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_hits: 1,
        persistent_prompt_cache_misses: 1,
        persistent_prompt_cache_tokens_saved: 2_048,
        persistent_prompt_cache_block_token_count: 2_048,
        persistent_prompt_cache_sequence_state_block_count: 2,
        persistent_prompt_cache_boundary_state_snapshot_count: 1,
        persistent_prompt_cache_visual_embedding_count: 0,
        persistent_prompt_cache_total_size_bytes: 40_000,
        persistent_prompt_cache_visual_embedding_total_size_bytes: 0,
        persistent_prompt_cache_maximum_size_bytes: 50_000,
        persistent_prompt_cache_visual_embedding_hits: 0,
        persistent_prompt_cache_visual_embedding_misses: 0,
        persistent_prompt_cache_visual_embedding_rows_loaded: 0,
    };
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(4_096),
        ScriptedChatEngine::with_cached_token_count(2_048)
            .with_active_generation_prompt_cache_stats(prompt_cache_stats_after_lookup.clone()),
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(91, 32)))
        .await
        .expect("the worker should receive a chat request");

    assert_eq!(
        next_event(&mut supervisor_reader).await,
        prompt_cache_stats_after_lookup,
        "the supervisor must see the classified cache lookup before a long prefill completes"
    );
    close_worker_transport(supervisor_writer, worker_task).await;
}
