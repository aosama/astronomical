use std::{fs, path::Path, time::Duration};

use astronomical_supervisor::{ChatGenerationExecutor, ChatGenerationStreamEvent};
use tokio::time::timeout;

use super::model_prefill_benchmark_report::PrefillMeasurementAccumulator;
use super::model_prefill_optimizer_comparison::benchmark_command;
use super::model_prefill_qualification_worker::{
    build_prefill_qualification_prompt, configured_prefill_qualification_model_directory,
    launch_prepared_prefill_qualification_worker, prepare_prefill_qualification_worker,
    wait_until_prefill_qualification_worker_is_idle,
};

const OPTIMIZER_CANDIDATE_OBSERVATION_PROMPT_TOKENS: usize = 20_000;
const OPTIMIZER_CANDIDATE_OBSERVATION_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ornith through the configuration-enabled optimizer and observes large candidates"]
async fn should_observe_complete_large_candidates_with_the_configuration_enabled_optimizer() {
    timeout(
        OPTIMIZER_CANDIDATE_OBSERVATION_TIMEOUT,
        run_configuration_enabled_optimizer_candidate_observation(),
    )
    .await
    .expect("the optimizer candidate observation must finish within 115 seconds");
}

async fn run_configuration_enabled_optimizer_candidate_observation() {
    let configured_model_directory = configured_prefill_qualification_model_directory();
    let optimizer_state_reload_is_required =
        std::env::var_os("ASTRONOMICAL_REQUIRE_PREFILL_OPTIMIZER_STATE_RELOAD").is_some();
    let mut exact_prompt_content = build_prefill_qualification_prompt(
        &configured_model_directory,
        OPTIMIZER_CANDIDATE_OBSERVATION_PROMPT_TOKENS,
    );
    if let Ok(prompt_variant) = std::env::var("ASTRONOMICAL_PREFILL_OPTIMIZER_PROMPT_VARIANT") {
        exact_prompt_content =
            format!("Qualification variant: {prompt_variant}\n\n{exact_prompt_content}");
    }
    let prepared_prefill_qualification_worker =
        prepare_prefill_qualification_worker(&configured_model_directory, None, None);
    if optimizer_state_reload_is_required {
        assert!(
            prepared_prefill_qualification_worker.optimizer_state_loaded_before_run,
            "the second production-worker run must discover optimizer state from the first run"
        );
    }
    let worker_handle = launch_prepared_prefill_qualification_worker(
        &prepared_prefill_qualification_worker,
        &configured_model_directory,
    )
    .await;
    wait_until_prefill_qualification_worker_is_idle(&worker_handle, "optimizer-candidates").await;
    let mut stream_receiver = worker_handle
        .start_chat_generation(benchmark_command(exact_prompt_content, 1))
        .await
        .expect("the optimizer candidate observation should start");
    let mut prefill_measurements = PrefillMeasurementAccumulator::new();
    let mut completion_was_observed = false;

    while let Some(stream_event) = stream_receiver.recv().await {
        match stream_event {
            ChatGenerationStreamEvent::PrefillProgress {
                processed_tokens,
                total_tokens,
                elapsed_millis,
                forward_prefill_chunck_elapsed_millis,
                completed_prefill_chunck_tokens,
                mlx_active_memory_bytes,
                mlx_allocator_cache_memory_bytes,
                mlx_peak_memory_bytes,
            } => {
                assert!(
                    total_tokens >= OPTIMIZER_CANDIDATE_OBSERVATION_PROMPT_TOKENS as u32,
                    "the optimizer candidate observation prompt is shorter than requested"
                );
                prefill_measurements.record(
                    processed_tokens,
                    elapsed_millis,
                    forward_prefill_chunck_elapsed_millis,
                    completed_prefill_chunck_tokens,
                    mlx_active_memory_bytes,
                    mlx_allocator_cache_memory_bytes,
                    mlx_peak_memory_bytes,
                );
                eprintln!(
                    "[prefill-optimizer:optimizer-candidates] processed={processed_tokens}/{total_tokens} completed_prefill_chunck_tokens={completed_prefill_chunck_tokens:?}"
                );
            }
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                ..
            } => {
                assert!(
                    prompt_token_count >= OPTIMIZER_CANDIDATE_OBSERVATION_PROMPT_TOKENS as u32,
                    "the completed optimizer observation reports fewer tokens than requested"
                );
                assert_eq!(generated_token_count, 1);
                assert_eq!(cached_token_count, 0);
                completion_was_observed = true;
                break;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the optimizer candidate observation failed: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the optimizer candidate observation stream failed: {error_code:?}");
            }
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. } => {}
        }
    }

    assert!(completion_was_observed);
    assert_cumulative_latency_optimizer_evidence(&prefill_measurements);
    assert_persisted_requested_action_transitions(
        &prepared_prefill_qualification_worker.optimizer_state_file_path(),
    );
    worker_handle
        .shutdown()
        .await
        .expect("the optimizer candidate observation worker should terminate");
}

pub(super) fn assert_cumulative_latency_optimizer_evidence(
    prefill_measurements: &PrefillMeasurementAccumulator,
) {
    assert!(!prefill_measurements.chuncks().is_empty());
    assert!(prefill_measurements.cumulative_elapsed_millis() > 0);
    assert!(
        prefill_measurements
            .chuncks()
            .iter()
            .all(|prefill_chunck_measurement| {
                prefill_chunck_measurement.actual_prefill_chunck_tokens
                    == prefill_chunck_measurement.reported_completed_prefill_chunck_tokens
            }),
        "the progress protocol should report actual completed advancement without calling it the selected request"
    );
}

fn assert_persisted_requested_action_transitions(optimizer_state_file_path: &Path) {
    let serialized_optimizer_state = fs::read_to_string(optimizer_state_file_path)
        .expect("the configuration-enabled optimizer should persist transition state");
    let optimizer_state: serde_json::Value = serde_json::from_str(&serialized_optimizer_state)
        .expect("the persisted optimizer state should be valid JSON");
    assert_eq!(optimizer_state["format_version"], 4);
    let context_buckets = optimizer_state["context_buckets"]
        .as_array()
        .expect("optimizer context buckets should be an array");
    let persisted_transition_count = context_buckets
        .iter()
        .flat_map(|context_bucket| {
            context_bucket["candidates"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter_map(|candidate_statistics| candidate_statistics["observations"].as_array())
        .map(Vec::len)
        .sum::<usize>();
    assert!(
        persisted_transition_count > 0,
        "optimizer state should retain requested-action transition outcomes"
    );
}
