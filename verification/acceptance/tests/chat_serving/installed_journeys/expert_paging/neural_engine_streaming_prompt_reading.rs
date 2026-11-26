//! Neural Engine prompt reading on the streaming engine (issue #2585): a
//! dense model with every layer kept in memory builds its programs beside
//! itself, attaches them between requests, and reads a long prompt a span
//! at a time on both engines.
//!
//! One case with the programs and one without, a process each, on the test
//! run's `model_directory` and `paging_experiment` limit. The limit must
//! hold the whole model and its programs, or the programs are refused and
//! the case fails. Building the programs takes about five minutes when macOS
//! holds no prepared copy, and their memory is measured after half a minute
//! of an idle GPU.
//!
//! `cargo measure-neural-engine-streaming-prompt-reading --configuration PATH`.
//! In each `phase=long_prompt_read_` line, `request_elapsed_seconds` less
//! `generation_elapsed_seconds` is the time to the first token.

use ipc_protocol::NeuralEnginePrefillStatus;
use model_serving::InferenceEngine;
use serial_test::serial;
use tokio::time::{Duration, Instant, sleep, timeout};

use super::reply_speed_cases::serve_warmup_and_read_a_long_prompt;
use crate::serving_acceptance::support::performance_attribution::{
    StreamingJourneyOptions, load_paging_experiment_engine_with,
};

const CASE_TIMEOUT: Duration = Duration::from_secs(540);
const PROGRAMS_TIMEOUT: Duration = Duration::from_secs(450);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// An idle worker steps the preparation this often. The preparation takes a
/// gap of a second between steps for work nobody announced and starts its
/// idle wait again, so a slower step would never let it measure.
const IDLE_STEP_INTERVAL: Duration = Duration::from_millis(100);

#[tokio::test]
#[ignore = "builds Neural Engine programs beside a dense model on the streaming engine"]
#[serial]
async fn should_read_a_long_prompt_on_the_streaming_engine_with_neural_engine_programs() {
    read_a_long_prompt(true, "long_prompt_read_with_neural_engine_programs").await;
}

#[tokio::test]
#[ignore = "reads the same prompt on the streaming engine's GPU alone"]
#[serial]
async fn should_read_a_long_prompt_on_the_streaming_engine_with_the_gpu_alone() {
    read_a_long_prompt(false, "long_prompt_read_with_the_gpu_alone").await;
}

async fn read_a_long_prompt(reads_prompts_on_the_neural_engine: bool, phase_name: &str) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let journey_options = StreamingJourneyOptions {
        reads_prompts_on_the_neural_engine,
        ..StreamingJourneyOptions::default()
    };
    let read_within_budget = timeout(CASE_TIMEOUT, async {
        let mut paging = load_paging_experiment_engine_with(
            "neural_engine_streaming_model_load",
            journey_options,
        )
        .await;
        if reads_prompts_on_the_neural_engine {
            let Some(ipc_protocol::WorkerEvent::NeuralEnginePrefillChanged {
                neural_engine_prefill: status_at_load,
            }) = paging.neural_engine_prefill_at_load.take()
            else {
                panic!("the load reported nothing about Neural Engine prompt reading");
            };
            wait_for_the_programs(&mut paging.engine, status_at_load).await;
        }
        let served_model = (&*paging.model_directory, &*paging.end_of_sequence_token_ids);
        let _ =
            serve_warmup_and_read_a_long_prompt(&mut paging.engine, served_model, phase_name).await;
    });
    assert!(
        read_within_budget.await.is_ok(),
        "{phase_name}: the programs and the prompt took longer than the case allows"
    );
}

/// Advances the preparation as an idle worker does, every 100 ms, reporting
/// progress and time left every five seconds, until the programs are
/// attached.
async fn wait_for_the_programs(
    engine: &mut impl InferenceEngine,
    status_at_load: NeuralEnginePrefillStatus,
) {
    let started_at = Instant::now();
    let mut latest_status = status_at_load;
    let mut next_progress_report_at = started_at;
    while started_at.elapsed() < PROGRAMS_TIMEOUT {
        if let Some(reported_status) = engine
            .advance_neural_engine_preparation()
            .await
            .expect("the preparation advances")
        {
            latest_status = reported_status;
        }
        match &latest_status {
            NeuralEnginePrefillStatus::Active { .. } => {
                eprintln!(
                    "[neural-engine-streaming] status=attached elapsed_seconds={:.0} {latest_status:?}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            NeuralEnginePrefillStatus::Preparing { .. } => {
                if Instant::now() >= next_progress_report_at {
                    next_progress_report_at = Instant::now() + PROGRESS_INTERVAL;
                    report_preparation_progress(&latest_status, started_at);
                }
            }
            other_status => panic!("the programs were not prepared: {other_status:?}"),
        }
        sleep(IDLE_STEP_INTERVAL).await;
    }
    panic!("the programs did not attach in time: {latest_status:?}");
}

fn report_preparation_progress(preparing_status: &NeuralEnginePrefillStatus, started_at: Instant) {
    let NeuralEnginePrefillStatus::Preparing {
        completed_program_count,
        total_program_count,
        estimated_remaining_ms,
        is_waiting_to_measure,
        ..
    } = preparing_status
    else {
        return;
    };
    eprintln!(
        "[neural-engine-streaming] status=preparing programs={completed_program_count}/{total_program_count} progress_percent={} elapsed_seconds={:.0} eta_seconds={} waiting_to_measure={is_waiting_to_measure}",
        completed_program_count * 100 / (*total_program_count).max(1),
        started_at.elapsed().as_secs_f64(),
        estimated_remaining_ms.map_or_else(|| "unknown".to_owned(), |ms| (ms / 1000).to_string()),
    );
}
