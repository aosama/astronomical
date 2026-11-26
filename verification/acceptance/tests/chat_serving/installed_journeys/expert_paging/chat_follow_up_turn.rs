//! How long each text engine takes to the first token of a conversation's
//! next turn, under the same `paging_experiment` limit (issue #2585).
//!
//! A fresh prompt, which every other reply journey measures, never shows
//! this: the engine has just read all but the new message. The resident
//! engine adopts the prompt it kept in memory; the streaming engine restores
//! it from the `astro_ssd_prompt_cache`. Both get a store in a temporary
//! folder, as serving gives them one by default. A warm-up, then a
//! 10,000-token prompt with 16 tokens written, then the next turn: that
//! prompt, the reply and a 40-token message.
//!
//! `cargo measure-chat-follow-up-turn --configuration PATH` runs the
//! resident case and then the streaming one, a process each. In each
//! `phase=..._follow_up_turn` line, `request_elapsed_seconds` less
//! `generation_elapsed_seconds` is the time to the first token.

use model_serving::AstroSsdPromptCacheDiskStoreConfig;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use super::reply_speed_cases::{read_the_follow_up_turn, serve_warmup_and_read_a_long_prompt};
use crate::serving_acceptance::support::performance_attribution::{
    load_configured_resident_engine_keeping_prompts, load_paging_experiment_engine_keeping_prompts,
};

const CASE_TIMEOUT: Duration = Duration::from_secs(150);
fn store_in(store_directory: &tempfile::TempDir) -> Option<AstroSsdPromptCacheDiskStoreConfig> {
    Some(AstroSsdPromptCacheDiskStoreConfig::new(
        store_directory.path().to_path_buf(),
        store_directory.path().to_path_buf(),
        crate::common::configured_model_artifact_astro_ssd_prompt_cache_maximum_size_bytes(),
    ))
}

#[tokio::test]
#[ignore = "reads a conversation's next turn on the resident engine"]
#[serial]
async fn should_measure_a_chat_follow_up_turn_on_the_resident_engine() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let store_directory = tempfile::tempdir().expect("a store folder is made");
    timeout(CASE_TIMEOUT, async {
        let mut loaded = load_configured_resident_engine_keeping_prompts(
            "follow_up_resident_model_load",
            store_in(&store_directory),
        )
        .await;
        let served_model = (&*loaded.model_directory, &*loaded.end_of_sequence_token_ids);
        let phase_name = "chat_read_by_the_resident_engine";
        let written_token_ids =
            serve_warmup_and_read_a_long_prompt(&mut loaded.engine, served_model, phase_name).await;
        read_the_follow_up_turn(
            &mut loaded.engine,
            served_model,
            &written_token_ids,
            phase_name,
        )
        .await;
    })
    .await
    .expect("the follow-up case must finish within its budget");
}

#[tokio::test]
#[ignore = "reads a conversation's next turn on the streaming engine"]
#[serial]
async fn should_measure_a_chat_follow_up_turn_on_the_streaming_engine() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let store_directory = tempfile::tempdir().expect("a store folder is made");
    timeout(CASE_TIMEOUT, async {
        let mut paging = load_paging_experiment_engine_keeping_prompts(
            "follow_up_streaming_model_load",
            store_in(&store_directory),
        )
        .await;
        let served_model = (&*paging.model_directory, &*paging.end_of_sequence_token_ids);
        let phase_name = "chat_read_by_the_streaming_engine";
        let written_token_ids =
            serve_warmup_and_read_a_long_prompt(&mut paging.engine, served_model, phase_name).await;
        read_the_follow_up_turn(
            &mut paging.engine,
            served_model,
            &written_token_ids,
            phase_name,
        )
        .await;
    })
    .await
    .expect("the follow-up case must finish within its budget");
}
