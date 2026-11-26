use super::http_transport;
use super::live_progress;
use super::*;

/// Parameterizes the E2E `astro_ssd_prompt_cache` stats test so the same runner covers both the
/// representative Romeo and Juliet scenarios without duplicating logic.
pub(super) struct AstroSsdPromptCacheStatsE2eCase {
    /// Short label used in all `eprintln!` log lines, e.g. `"2k"` or `"50k"`.
    test_name: &'static str,
    /// The full prompt text sent to the model as the user message.
    prompt: &'static str,
    /// Word count of the prompt, for log labels only.
    prompt_word_count: usize,
    /// Maximum output tokens requested. Kept small so generation finishes fast;
    /// the test cares about prefill and `astro_ssd_prompt_cache` behavior, not generation quality.
    maximum_output_tokens: u16,
    /// Wall-clock timeout for the entire test (model load + two requests).
    timeout: Duration,
}

pub(super) fn five_thousand_word_case() -> AstroSsdPromptCacheStatsE2eCase {
    AstroSsdPromptCacheStatsE2eCase {
        test_name: "romeo-and-juliet-5k",
        prompt: FIVE_THOUSAND_WORD_ROMEO_AND_JULIET_PROMPT,
        prompt_word_count: 5_000,
        maximum_output_tokens: 16,
        timeout: Duration::from_secs(115),
    }
}

pub(super) async fn run_astro_ssd_prompt_cache_stats_e2e_with_timeout(
    astro_ssd_prompt_cache_stats_e2e_case: AstroSsdPromptCacheStatsE2eCase,
) {
    let test_name = astro_ssd_prompt_cache_stats_e2e_case.test_name;
    let log_prefix = format!("[e2e-astro-ssd-prompt-cache-stats:{test_name}]");
    let started_at = Instant::now();
    let timeout_deadline = sleep(astro_ssd_prompt_cache_stats_e2e_case.timeout);
    let mut progress_interval = interval(Duration::from_secs(PROGRESS_INTERVAL_SECONDS));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::pin!(timeout_deadline);
    let test_future =
        run_astro_ssd_prompt_cache_stats_e2e(&astro_ssd_prompt_cache_stats_e2e_case, &log_prefix);

    eprintln!(
        "{log_prefix} status=start timeout_seconds={} prompt_word_count={}",
        astro_ssd_prompt_cache_stats_e2e_case.timeout.as_secs(),
        astro_ssd_prompt_cache_stats_e2e_case.prompt_word_count
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
                    "{log_prefix} the persistent astro-ssd-prompt-cache E2E test exceeded {} seconds",
                    astro_ssd_prompt_cache_stats_e2e_case.timeout.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = astro_ssd_prompt_cache_stats_e2e_case.timeout.saturating_sub(elapsed);
                eprintln!(
                    "{log_prefix} status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

async fn run_astro_ssd_prompt_cache_stats_e2e(
    astro_ssd_prompt_cache_stats_e2e_case: &AstroSsdPromptCacheStatsE2eCase,
    log_prefix: &str,
) {
    let (
        isolated_worker_home,
        astro_ssd_prompt_cache_directory_path,
        worker_executable_path,
        configured_model_directory,
        model_policy_catalog,
        worker_startup_configuration,
    ) = create_astro_ssd_prompt_cache_stats_worker_configuration();

    eprintln!(
        "{log_prefix} launching the model-artifact worker from isolated config.json; astro_ssd_prompt_cache_dir={}",
        astro_ssd_prompt_cache_directory_path.display()
    );
    let running_server = start_stats_e2e_server(
        log_prefix,
        isolated_worker_home.path(),
        worker_executable_path,
        model_policy_catalog,
        worker_startup_configuration,
        &configured_model_directory,
    )
    .await;
    let server_address = running_server.server_address;

    live_progress::wait_until_ready(server_address, log_prefix).await;

    let astro_ssd_prompt_cache_block_token_count = run_first_request_expecting_miss(
        astro_ssd_prompt_cache_stats_e2e_case,
        log_prefix,
        server_address,
        &astro_ssd_prompt_cache_directory_path,
    )
    .await;
    run_second_request_expecting_hit(
        astro_ssd_prompt_cache_stats_e2e_case,
        log_prefix,
        server_address,
        &astro_ssd_prompt_cache_directory_path,
        astro_ssd_prompt_cache_block_token_count,
    )
    .await;

    stop_stats_e2e_server(running_server, log_prefix).await;

    eprintln!(
        "{log_prefix} astro SSD prompt cache miss then hit verified through /v1/ssd-prompt-cache/stats"
    );
}

/// The HTTP server and the real worker behind it, kept together because they
/// stop in one order: the server task ends first, then the worker is reaped.
struct RunningStatsE2eServer {
    server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
    worker_handle: WorkerHandle,
}

async fn start_stats_e2e_server(
    log_prefix: &str,
    isolated_worker_home: &std::path::Path,
    worker_executable_path: std::path::PathBuf,
    model_policy_catalog: std::sync::Arc<
        std::collections::HashMap<String, astronomical_daemon::RuntimeModelPolicy>,
    >,
    worker_startup_configuration: ipc_protocol::WorkerStartupConfiguration,
    configured_model_directory: &std::path::Path,
) -> RunningStatsE2eServer {
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(isolated_worker_home.join("logs").as_path())
            .expect("performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the daemon should launch the model-artifact worker");

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
                model_id(),
                configured_model_directory,
                20_480,
            )],
        ),
    )
    .with_graceful_shutdown(async {
        let _ = shutdown_receiver.await;
    });
    let server_task = tokio::spawn(async move { server.await });
    RunningStatsE2eServer {
        server_address,
        shutdown_sender,
        server_task,
        worker_handle,
    }
}

async fn stop_stats_e2e_server(running_server: RunningStatsE2eServer, log_prefix: &str) {
    let _ = running_server.shutdown_sender.send(());
    running_server
        .server_task
        .await
        .expect("the E2E HTTP server task should not panic")
        .expect("the E2E HTTP server should stop cleanly");
    eprintln!("{log_prefix} shutting down the real worker");
    running_server
        .worker_handle
        .shutdown()
        .await
        .expect("the real worker should terminate and be reaped");
}

/// Posts the scenario prompt for one phase and requires a clean 200 stream.
/// `phase_label` names the phase in log lines and `ordinal_label` ("first",
/// "second") names the request in assertion messages.
async fn post_phase_request_and_assert_clean_stream(
    server_address: SocketAddr,
    astro_ssd_prompt_cache_stats_e2e_case: &AstroSsdPromptCacheStatsE2eCase,
    log_prefix: &str,
    phase_label: &str,
    ordinal_label: &str,
    phase_started_at: Instant,
) {
    let chat_response = live_progress::post_chat_completion_with_live_progress(
        server_address,
        astro_ssd_prompt_cache_stats_e2e_case.prompt,
        astro_ssd_prompt_cache_stats_e2e_case.maximum_output_tokens,
        log_prefix,
        phase_label,
    )
    .await;
    eprintln!(
        "{log_prefix} {phase_label}: HTTP response fully received in {:.1}s, {} bytes",
        phase_started_at.elapsed().as_secs_f64(),
        chat_response.len()
    );
    assert!(
        chat_response.starts_with("HTTP/1.1 200 OK"),
        "the {ordinal_label} chat response should be 200 OK, got: {chat_response}"
    );
    assert!(
        chat_response.contains("data: [DONE]"),
        "the {ordinal_label} chat stream should finish cleanly: {chat_response}"
    );
}

/// Returns the astro SSD prompt cache block token count the second request is
/// later held to.
async fn run_first_request_expecting_miss(
    astro_ssd_prompt_cache_stats_e2e_case: &AstroSsdPromptCacheStatsE2eCase,
    log_prefix: &str,
    server_address: SocketAddr,
    astro_ssd_prompt_cache_directory_path: &std::path::Path,
) -> u64 {
    // ── Phase 1: first request — should be an `astro_ssd_prompt_cache` miss ──
    let phase_one_started_at = Instant::now();
    eprintln!(
        "{log_prefix} phase 1: sending first {}K-word request (expecting astro SSD prompt cache miss)",
        astro_ssd_prompt_cache_stats_e2e_case.prompt_word_count / 1_000
    );
    post_phase_request_and_assert_clean_stream(
        server_address,
        astro_ssd_prompt_cache_stats_e2e_case,
        log_prefix,
        "phase 1",
        "first",
        phase_one_started_at,
    )
    .await;
    let astro_ssd_prompt_cache_block_token_count = assert_stats_after_first_request_report_a_miss(
        server_address,
        log_prefix,
        phase_one_started_at,
    )
    .await;

    // Check what was saved to disk after phase 1.
    http_transport::log_astro_ssd_prompt_cache_directory_contents(
        astro_ssd_prompt_cache_directory_path,
        &format!(
            "{log_prefix} after phase 1 (expecting sequence state and boundary snapshot files)"
        ),
    );
    astro_ssd_prompt_cache_block_token_count
}

async fn assert_stats_after_first_request_report_a_miss(
    server_address: SocketAddr,
    log_prefix: &str,
    phase_one_started_at: Instant,
) -> u64 {
    let astro_ssd_prompt_cache_stats_after_first_request =
        http_transport::get_astro_ssd_prompt_cache_stats_json(server_address).await;
    eprintln!(
        "{log_prefix} phase 1: astro SSD prompt cache stats after first request = {}",
        serde_json::to_string_pretty(&astro_ssd_prompt_cache_stats_after_first_request)
            .unwrap_or_else(|_| "<serialization failed>".to_owned())
    );
    let astro_ssd_prompt_cache_misses_after_first_request =
        astro_ssd_prompt_cache_stats_after_first_request["astro_ssd_prompt_cache_misses"]
            .as_u64()
            .expect("the astro SSD prompt cache stats should report misses");
    let astro_ssd_prompt_cache_hits_after_first_request =
        astro_ssd_prompt_cache_stats_after_first_request["astro_ssd_prompt_cache_hits"]
            .as_u64()
            .expect("the astro SSD prompt cache stats should report hits");
    let astro_ssd_prompt_cache_block_token_count = astro_ssd_prompt_cache_stats_after_first_request
        ["astro_ssd_prompt_cache_block_token_count"]
        .as_u64()
        .expect("the astro SSD prompt cache stats should report the exact block token count");
    eprintln!(
        "{log_prefix} phase 1 complete in {:.1}s: hits={} misses={}",
        phase_one_started_at.elapsed().as_secs_f64(),
        astro_ssd_prompt_cache_hits_after_first_request,
        astro_ssd_prompt_cache_misses_after_first_request
    );
    assert!(
        astro_ssd_prompt_cache_misses_after_first_request >= 1,
        "the first request should have produced at least one astro SSD prompt cache miss"
    );
    assert_eq!(
        astro_ssd_prompt_cache_hits_after_first_request, 0,
        "the first request should not have produced any astro SSD prompt cache hits"
    );
    assert!(
        astro_ssd_prompt_cache_stats_after_first_request["astro_ssd_prompt_cache_sequence_state_block_count"]
            .as_u64()
            .is_some_and(|sequence_state_block_count| sequence_state_block_count > 0),
        "required astro-ssd-prompt-cache publication must be visible when the first response completes: {astro_ssd_prompt_cache_stats_after_first_request}"
    );
    astro_ssd_prompt_cache_block_token_count
}

/// The counters the second request is held to; the miss count is only logged.
struct StatsAfterSecondRequest {
    hits: u64,
    tokens_saved: u64,
    sequence_state_block_count: u64,
}

async fn run_second_request_expecting_hit(
    astro_ssd_prompt_cache_stats_e2e_case: &AstroSsdPromptCacheStatsE2eCase,
    log_prefix: &str,
    server_address: SocketAddr,
    astro_ssd_prompt_cache_directory_path: &std::path::Path,
    astro_ssd_prompt_cache_block_token_count: u64,
) {
    // ── Phase 2: same prompt — should be an `astro_ssd_prompt_cache` hit ──
    let phase_two_started_at = Instant::now();
    eprintln!(
        "{log_prefix} phase 2: sending the same {}K-word request (expecting astro SSD prompt cache hit)",
        astro_ssd_prompt_cache_stats_e2e_case.prompt_word_count / 1_000
    );
    post_phase_request_and_assert_clean_stream(
        server_address,
        astro_ssd_prompt_cache_stats_e2e_case,
        log_prefix,
        "phase 2",
        "second",
        phase_two_started_at,
    )
    .await;
    let stats_after_second_request = read_stats_after_second_request(
        server_address,
        log_prefix,
        astro_ssd_prompt_cache_directory_path,
        phase_two_started_at,
    )
    .await;
    assert_second_request_restored_the_prompt(
        &stats_after_second_request,
        astro_ssd_prompt_cache_block_token_count,
    );
}

async fn read_stats_after_second_request(
    server_address: SocketAddr,
    log_prefix: &str,
    astro_ssd_prompt_cache_directory_path: &std::path::Path,
    phase_two_started_at: Instant,
) -> StatsAfterSecondRequest {
    let astro_ssd_prompt_cache_stats_after_second_request =
        http_transport::get_astro_ssd_prompt_cache_stats_json(server_address).await;
    eprintln!(
        "{log_prefix} phase 2: astro SSD prompt cache stats after second request = {}",
        serde_json::to_string_pretty(&astro_ssd_prompt_cache_stats_after_second_request)
            .unwrap_or_else(|_| "<serialization failed>".to_owned())
    );
    let astro_ssd_prompt_cache_hits_after_second_request =
        astro_ssd_prompt_cache_stats_after_second_request["astro_ssd_prompt_cache_hits"]
            .as_u64()
            .expect("the astro SSD prompt cache stats should report hits after the second request");
    let astro_ssd_prompt_cache_tokens_saved_after_second_request =
        astro_ssd_prompt_cache_stats_after_second_request["astro_ssd_prompt_cache_tokens_saved"]
            .as_u64()
            .expect("the astro SSD prompt cache stats should report tokens saved after the second request");
    let astro_ssd_prompt_cache_sequence_state_block_count_after_second_request =
        astro_ssd_prompt_cache_stats_after_second_request["astro_ssd_prompt_cache_sequence_state_block_count"]
            .as_u64()
            .expect("the astro SSD prompt cache stats should report sequence-state block count");
    let astro_ssd_prompt_cache_misses_after_second_request =
        astro_ssd_prompt_cache_stats_after_second_request["astro_ssd_prompt_cache_misses"]
            .as_u64()
            .expect(
                "the astro SSD prompt cache stats should report misses after the second request",
            );
    eprintln!(
        "{log_prefix} phase 2 complete in {:.1}s: hits={} misses={} tokens_saved={} sequence_state_blocks={}",
        phase_two_started_at.elapsed().as_secs_f64(),
        astro_ssd_prompt_cache_hits_after_second_request,
        astro_ssd_prompt_cache_misses_after_second_request,
        astro_ssd_prompt_cache_tokens_saved_after_second_request,
        astro_ssd_prompt_cache_sequence_state_block_count_after_second_request
    );

    // If phase 2 was a miss instead of a hit, gather diagnostics before failing.
    if astro_ssd_prompt_cache_hits_after_second_request == 0 {
        log_second_request_miss_diagnostics(
            server_address,
            log_prefix,
            astro_ssd_prompt_cache_directory_path,
            &astro_ssd_prompt_cache_stats_after_second_request,
            astro_ssd_prompt_cache_sequence_state_block_count_after_second_request,
        )
        .await;
    }
    StatsAfterSecondRequest {
        hits: astro_ssd_prompt_cache_hits_after_second_request,
        tokens_saved: astro_ssd_prompt_cache_tokens_saved_after_second_request,
        sequence_state_block_count:
            astro_ssd_prompt_cache_sequence_state_block_count_after_second_request,
    }
}

async fn log_second_request_miss_diagnostics(
    server_address: SocketAddr,
    log_prefix: &str,
    astro_ssd_prompt_cache_directory_path: &std::path::Path,
    astro_ssd_prompt_cache_stats_after_second_request: &serde_json::Value,
    sequence_state_block_count: u64,
) {
    eprintln!("{log_prefix} DIAGNOSTIC: phase 2 reported 0 hits — gathering diagnostic evidence");
    eprintln!(
        "{log_prefix} DIAGNOSTIC: full astro SSD prompt cache stats JSON = {}",
        serde_json::to_string_pretty(astro_ssd_prompt_cache_stats_after_second_request)
            .unwrap_or_default()
    );
    let status_response = http_transport::get_endpoint(server_address, "/v1/status").await;
    eprintln!("{log_prefix} DIAGNOSTIC: /v1/status = {status_response}");
    http_transport::log_astro_ssd_prompt_cache_directory_contents(
        astro_ssd_prompt_cache_directory_path,
        &format!("{log_prefix} DIAGNOSTIC: after phase 2 miss"),
    );
    let boundary_state_snapshot_count =
        astro_ssd_prompt_cache_stats_after_second_request["astro_ssd_prompt_cache_boundary_state_snapshot_count"]
            .as_u64()
            .unwrap_or(0);
    if sequence_state_block_count == 0 {
        eprintln!(
            "{log_prefix} DIAGNOSTIC: sequence_state_block_count is 0 — the astro SSD prompt cache save during phase 1 likely failed; check worker stderr for persistent astro-ssd-prompt-cache save warnings"
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

fn assert_second_request_restored_the_prompt(
    stats_after_second_request: &StatsAfterSecondRequest,
    astro_ssd_prompt_cache_block_token_count: u64,
) {
    assert!(
        stats_after_second_request.hits >= 1,
        "the second request should have produced at least one astro SSD prompt cache hit; \
         see DIAGNOSTIC lines above for the reason"
    );
    assert!(
        stats_after_second_request.tokens_saved >= astro_ssd_prompt_cache_block_token_count,
        "the astro SSD prompt cache hit should have restored at least {} tokens; \
         tokens_saved={}",
        astro_ssd_prompt_cache_block_token_count,
        stats_after_second_request.tokens_saved
    );
    assert!(
        stats_after_second_request.sequence_state_block_count >= 1,
        "the astro SSD prompt cache should contain at least one sequence-state block after the first request saved it; \
         sequence_state_block_count={}",
        stats_after_second_request.sequence_state_block_count
    );
}
