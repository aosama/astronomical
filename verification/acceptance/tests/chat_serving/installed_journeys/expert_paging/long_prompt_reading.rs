//! How fast the text engine reads one 10,000-token prompt under the
//! `paging_experiment` limit: the figure a person waits on before the first
//! token, which a short prompt does not show.
//!
//! The engine reads the prompt the way it serves, in its configured prompt
//! chunk, with a hosted drafter loaded and silent. A warm-up of 500 tokens
//! in and 50 out comes first; then the prompt is read once and 16 tokens are
//! written.
//!
//! `cargo measure-long-prompt-reading --configuration PATH` runs the case.
//! Run it twice and compare within a sitting; one run is not a result. In
//! the `status=success phase=long_prompt_read` line,
//! `request_elapsed_seconds` less `generation_elapsed_seconds` is the time
//! to the first token.

use model_serving::Qwen3_5TextEngineTestControls;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use super::reply_speed_cases::serve_warmup_and_read_a_long_prompt;
use crate::serving_acceptance::support::performance_attribution::load_paging_experiment_engine;

/// A dense model reads 10,000 tokens in about forty seconds, after its load
/// and warm-up.
const CASE_TIMEOUT: Duration = Duration::from_secs(150);

#[tokio::test]
#[ignore = "reads a 10,000-token prompt on the installed model"]
#[serial]
async fn should_measure_long_prompt_reading() {
    within_the_case_budget(read_with_the_drafter_silent()).await;
}

async fn within_the_case_budget(long_prompt_case: impl Future<Output = ()>) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    timeout(CASE_TIMEOUT, long_prompt_case)
        .await
        .expect("the long prompt case must finish within its budget");
}

/// A hosted drafter stays loaded and proposes nothing, so the few tokens
/// after the prompt are plain ones.
async fn read_with_the_drafter_silent() {
    let mut paging = load_paging_experiment_engine("long_prompt_model_load").await;
    paging
        .engine
        .set_hosted_drafter_writes_replies_for_tests(false)
        .await
        .expect("the engine takes the drafter's part");
    serve_warmup_and_read_a_long_prompt(
        &mut paging.engine,
        (&paging.model_directory, &paging.end_of_sequence_token_ids),
        "long_prompt_read",
    )
    .await;
}
