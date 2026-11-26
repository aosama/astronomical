//! How fast each text engine reads one 10,000-token prompt, under the same
//! `paging_experiment` limit (issue #2585): the figure a person waits on
//! before the first token, and the one gap between the engines that a short
//! prompt does not show.
//!
//! Each engine reads the prompt the way it serves: the resident engine in
//! its production chunks, the streaming engine in its configured streaming
//! chunk, a hosted drafter loaded and silent on both. A warm-up of 500
//! tokens in and 50 out comes first; then the prompt is read once and 16
//! tokens are written.
//!
//! `cargo measure-long-prompt-reading --configuration PATH` runs the
//! resident case and then the streaming one, a process each. Run it twice
//! and compare within a sitting; one run of either is not a result. In each
//! `status=success phase=long_prompt_read_by_...` line,
//! `request_elapsed_seconds` less `generation_elapsed_seconds` is the time
//! to the first token.

use model_serving::Qwen3_5StreamingEngineTestControls;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use super::reply_speed_cases::serve_warmup_and_read_a_long_prompt;
use crate::serving_acceptance::support::performance_attribution::{
    load_configured_resident_engine, load_paging_experiment_engine,
};

/// A dense model reads 10,000 tokens in about forty seconds, after its load
/// and warm-up.
const CASE_TIMEOUT: Duration = Duration::from_secs(150);

#[tokio::test]
#[ignore = "reads a 10,000-token prompt on the resident engine"]
#[serial]
async fn should_measure_long_prompt_reading_on_the_resident_engine() {
    within_the_case_budget(read_on_the_resident_engine()).await;
}

#[tokio::test]
#[ignore = "reads a 10,000-token prompt on the streaming engine"]
#[serial]
async fn should_measure_long_prompt_reading_on_the_streaming_engine() {
    within_the_case_budget(read_with_the_drafter_silent()).await;
}

async fn within_the_case_budget(long_prompt_case: impl Future<Output = ()>) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    timeout(CASE_TIMEOUT, long_prompt_case)
        .await
        .expect("the long prompt case must finish within its budget");
}

async fn read_on_the_resident_engine() {
    let mut loaded = load_configured_resident_engine("long_prompt_resident_model_load").await;
    let served_model = (&*loaded.model_directory, &*loaded.end_of_sequence_token_ids);
    let phase_name = "long_prompt_read_by_the_resident_engine";
    serve_warmup_and_read_a_long_prompt(&mut loaded.engine, served_model, phase_name).await;
}

/// A hosted drafter stays loaded and proposes nothing, so the few tokens
/// after the prompt are plain ones.
async fn read_with_the_drafter_silent() {
    let mut paging = load_paging_experiment_engine("long_prompt_streaming_model_load").await;
    paging
        .engine
        .set_hosted_drafter_writes_replies_for_tests(false)
        .await
        .expect("the engine takes the drafter's part");
    serve_warmup_and_read_a_long_prompt(
        &mut paging.engine,
        (&paging.model_directory, &paging.end_of_sequence_token_ids),
        "long_prompt_read_by_the_streaming_engine",
    )
    .await;
}
