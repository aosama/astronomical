use super::http_transport::{get_cache_stats_json, get_endpoint, log_cache_directory_contents};
use super::live_progress::{post_chat_completion_with_live_progress, wait_until_ready};
use super::*;

/// Parameterizes the E2E cache stats test so the same runner covers both the
/// representative Romeo and Juliet scenarios without duplicating logic.
pub(super) struct CacheStatsE2eCase {
    /// Short label used in all `eprintln!` log lines, e.g. `"2k"` or `"50k"`.
    test_name: &'static str,
    /// The full prompt text sent to the model as the user message.
    prompt: &'static str,
    /// Word count of the prompt, for log labels only.
    prompt_word_count: usize,
    /// Maximum output tokens requested. Kept small so generation finishes fast;
    /// the test cares about prefill and cache behavior, not generation quality.
    maximum_output_tokens: u16,
    /// Wall-clock timeout for the entire test (model load + two requests).
    timeout: Duration,
}

pub(super) fn five_thousand_word_case() -> CacheStatsE2eCase {
    CacheStatsE2eCase {
        test_name: "romeo-and-juliet-5k",
        prompt: FIVE_THOUSAND_WORD_ROMEO_AND_JULIET_PROMPT,
        prompt_word_count: 5_000,
        maximum_output_tokens: 16,
        timeout: Duration::from_secs(115),
    }
}

pub(super) async fn run_cache_stats_e2e_with_timeout(cache_stats_e2e_case: CacheStatsE2eCase) {
    let test_name = cache_stats_e2e_case.test_name;
    let log_prefix = format!("[e2e-cache-stats:{test_name}]");
    let started_at = Instant::now();
    let timeout_deadline = sleep(cache_stats_e2e_case.timeout);
    let mut progress_interval = interval(Duration::from_secs(PROGRESS_INTERVAL_SECONDS));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::pin!(timeout_deadline);
    let test_future = run_cache_stats_e2e(&cache_stats_e2e_case, &log_prefix);

    eprintln!(
        "{log_prefix} status=start timeout_seconds={} prompt_word_count={}",
        cache_stats_e2e_case.timeout.as_secs(),
        cache_stats_e2e_case.prompt_word_count
    );
    progress_interval.tick().await; // consume the immediate first tick

    tokio::pin!(test_future);
    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "{log_prefix} status=success elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!(
                    "{log_prefix} the persistent prompt-cache E2E test exceeded {} seconds",
                    cache_stats_e2e_case.timeout.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = cache_stats_e2e_case.timeout.saturating_sub(elapsed);
                eprintln!(
                    "{log_prefix} status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

async fn run_cache_stats_e2e(cache_stats_e2e_case: &CacheStatsE2eCase, log_prefix: &str) {
    let (
        isolated_worker_home,
        persistent_prompt_cache_directory_path,
        worker_executable_path,
        configured_model_directory,
        worker_startup_configuration,
    ) = create_cache_stats_worker_configuration();

    eprintln!(
        "{log_prefix} launching the model-artifact worker from isolated config.json; prompt_cache_dir={}",
        persistent_prompt_cache_directory_path.display()
    );
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(isolated_worker_home.path().join("logs").as_path())
            .expect("performance log should open"),
        single_model_directories(MODEL_ID, &configured_model_directory),
        20_480,
        worker_startup_configuration,
    )
    .await
    .expect("the supervisor should launch the model-artifact worker");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the E2E HTTP listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the E2E HTTP listener should expose its address");
    eprintln!("{log_prefix} HTTP server bound on {server_address}");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let server = axum::serve(
        listener,
        build_application_with_discovered_models(
            worker_handle.clone(),
            vec![discovered_model_artifact(
                MODEL_ID,
                &configured_model_directory,
                20_480,
            )],
        ),
    )
    .with_graceful_shutdown(async {
        let _ = shutdown_receiver.await;
    });
    let server_task = tokio::spawn(async move { server.await });

    wait_until_ready(server_address, log_prefix).await;

    // ── Phase 1: first request — should be a cache miss ──
    let phase_one_started_at = Instant::now();
    eprintln!(
        "{log_prefix} phase 1: sending first {}K-word request (expecting cache miss)",
        cache_stats_e2e_case.prompt_word_count / 1_000
    );
    let first_chat_response = post_chat_completion_with_live_progress(
        server_address,
        cache_stats_e2e_case.prompt,
        cache_stats_e2e_case.maximum_output_tokens,
        log_prefix,
        "phase 1",
    )
    .await;
    eprintln!(
        "{log_prefix} phase 1: HTTP response fully received in {:.1}s, {} bytes",
        phase_one_started_at.elapsed().as_secs_f64(),
        first_chat_response.len()
    );
    assert!(
        first_chat_response.starts_with("HTTP/1.1 200 OK"),
        "the first chat response should be 200 OK, got: {first_chat_response}"
    );
    assert!(
        first_chat_response.contains("data: [DONE]"),
        "the first chat stream should finish cleanly: {first_chat_response}"
    );

    let cache_stats_after_first_request = get_cache_stats_json(server_address).await;
    eprintln!(
        "{log_prefix} phase 1: cache stats after first request = {}",
        serde_json::to_string_pretty(&cache_stats_after_first_request)
            .unwrap_or_else(|_| "<serialization failed>".to_owned())
    );
    let persistent_prompt_cache_misses_after_first_request =
        cache_stats_after_first_request["persistent_prompt_cache_misses"]
            .as_u64()
            .expect("the cache stats should report misses");
    let persistent_prompt_cache_hits_after_first_request =
        cache_stats_after_first_request["persistent_prompt_cache_hits"]
            .as_u64()
            .expect("the cache stats should report hits");
    let persistent_prompt_cache_block_token_count =
        cache_stats_after_first_request["persistent_prompt_cache_block_token_count"]
            .as_u64()
            .expect("the cache stats should report the exact block token count");
    eprintln!(
        "{log_prefix} phase 1 complete in {:.1}s: hits={} misses={}",
        phase_one_started_at.elapsed().as_secs_f64(),
        persistent_prompt_cache_hits_after_first_request,
        persistent_prompt_cache_misses_after_first_request
    );
    assert!(
        persistent_prompt_cache_misses_after_first_request >= 1,
        "the first request should have produced at least one cache miss"
    );
    assert_eq!(
        persistent_prompt_cache_hits_after_first_request, 0,
        "the first request should not have produced any cache hits"
    );
    assert!(
        cache_stats_after_first_request["persistent_prompt_cache_sequence_state_block_count"]
            .as_u64()
            .is_some_and(|sequence_state_block_count| sequence_state_block_count > 0),
        "required prompt-cache publication must be visible when the first response completes: {cache_stats_after_first_request}"
    );

    // Check what was saved to disk after phase 1.
    log_cache_directory_contents(
        &persistent_prompt_cache_directory_path,
        &format!(
            "{log_prefix} after phase 1 (expecting sequence state and boundary snapshot files)"
        ),
    );

    // ── Phase 2: same prompt — should be a cache hit ──
    let phase_two_started_at = Instant::now();
    eprintln!(
        "{log_prefix} phase 2: sending the same {}K-word request (expecting cache hit)",
        cache_stats_e2e_case.prompt_word_count / 1_000
    );
    let second_chat_response = post_chat_completion_with_live_progress(
        server_address,
        cache_stats_e2e_case.prompt,
        cache_stats_e2e_case.maximum_output_tokens,
        log_prefix,
        "phase 2",
    )
    .await;
    eprintln!(
        "{log_prefix} phase 2: HTTP response fully received in {:.1}s, {} bytes",
        phase_two_started_at.elapsed().as_secs_f64(),
        second_chat_response.len()
    );
    assert!(
        second_chat_response.starts_with("HTTP/1.1 200 OK"),
        "the second chat response should be 200 OK, got: {second_chat_response}"
    );
    assert!(
        second_chat_response.contains("data: [DONE]"),
        "the second chat stream should finish cleanly: {second_chat_response}"
    );

    let cache_stats_after_second_request = get_cache_stats_json(server_address).await;
    eprintln!(
        "{log_prefix} phase 2: cache stats after second request = {}",
        serde_json::to_string_pretty(&cache_stats_after_second_request)
            .unwrap_or_else(|_| "<serialization failed>".to_owned())
    );
    let persistent_prompt_cache_hits_after_second_request =
        cache_stats_after_second_request["persistent_prompt_cache_hits"]
            .as_u64()
            .expect("the cache stats should report hits after the second request");
    let persistent_prompt_cache_tokens_saved_after_second_request =
        cache_stats_after_second_request["persistent_prompt_cache_tokens_saved"]
            .as_u64()
            .expect("the cache stats should report tokens saved after the second request");
    let persistent_prompt_cache_sequence_state_block_count_after_second_request =
        cache_stats_after_second_request["persistent_prompt_cache_sequence_state_block_count"]
            .as_u64()
            .expect("the cache stats should report sequence-state block count");
    let persistent_prompt_cache_misses_after_second_request =
        cache_stats_after_second_request["persistent_prompt_cache_misses"]
            .as_u64()
            .expect("the cache stats should report misses after the second request");
    eprintln!(
        "{log_prefix} phase 2 complete in {:.1}s: hits={} misses={} tokens_saved={} sequence_state_blocks={}",
        phase_two_started_at.elapsed().as_secs_f64(),
        persistent_prompt_cache_hits_after_second_request,
        persistent_prompt_cache_misses_after_second_request,
        persistent_prompt_cache_tokens_saved_after_second_request,
        persistent_prompt_cache_sequence_state_block_count_after_second_request
    );

    // If phase 2 was a miss instead of a hit, gather diagnostics before failing.
    if persistent_prompt_cache_hits_after_second_request == 0 {
        eprintln!(
            "{log_prefix} DIAGNOSTIC: phase 2 reported 0 hits — gathering diagnostic evidence"
        );
        eprintln!(
            "{log_prefix} DIAGNOSTIC: full cache stats JSON = {}",
            serde_json::to_string_pretty(&cache_stats_after_second_request).unwrap_or_default()
        );
        let status_response = get_endpoint(server_address, "/v1/status").await;
        eprintln!("{log_prefix} DIAGNOSTIC: /v1/status = {status_response}");
        log_cache_directory_contents(
            &persistent_prompt_cache_directory_path,
            &format!("{log_prefix} DIAGNOSTIC: after phase 2 miss"),
        );
        let sequence_state_block_count =
            persistent_prompt_cache_sequence_state_block_count_after_second_request;
        let boundary_state_snapshot_count =
            cache_stats_after_second_request["persistent_prompt_cache_boundary_state_snapshot_count"]
                .as_u64()
                .unwrap_or(0);
        if sequence_state_block_count == 0 {
            eprintln!(
                "{log_prefix} DIAGNOSTIC: sequence_state_block_count is 0 — the cache save during phase 1 likely failed; check worker stderr for persistent prompt-cache save warnings"
            );
        } else if boundary_state_snapshot_count == 0 {
            eprintln!(
                "{log_prefix} DIAGNOSTIC: sequence_state_block_count={sequence_state_block_count} but boundary_state_snapshot_count=0 — the boundary state was not saved; restore requires both file families"
            );
        } else {
            eprintln!(
                "{log_prefix} DIAGNOSTIC: sequence_state_block_count={sequence_state_block_count} boundary_state_snapshot_count={boundary_state_snapshot_count} — files exist on disk but lookup returned a miss; the prompt may not hash to the same chain or the boundary snapshot walk failed"
            );
        }
    }

    assert!(
        persistent_prompt_cache_hits_after_second_request >= 1,
        "the second request should have produced at least one cache hit; \
         see DIAGNOSTIC lines above for the reason"
    );
    assert!(
        persistent_prompt_cache_tokens_saved_after_second_request
            >= persistent_prompt_cache_block_token_count,
        "the cache hit should have restored at least {} tokens; \
         tokens_saved={persistent_prompt_cache_tokens_saved_after_second_request}",
        persistent_prompt_cache_block_token_count
    );
    assert!(
        persistent_prompt_cache_sequence_state_block_count_after_second_request >= 1,
        "the cache should contain at least one sequence-state block after the first request saved it; \
         sequence_state_block_count={persistent_prompt_cache_sequence_state_block_count_after_second_request}"
    );

    let _ = shutdown_sender.send(());
    server_task
        .await
        .expect("the E2E HTTP server task should not panic")
        .expect("the E2E HTTP server should stop cleanly");
    eprintln!("{log_prefix} shutting down the real worker");
    worker_handle
        .shutdown()
        .await
        .expect("the real worker should terminate and be reaped");

    eprintln!("{log_prefix} cache miss then cache hit verified through /v1/cache/stats");
}
