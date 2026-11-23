//! A reply must be the same tokens whichever way its missing experts arrive.
//!
//! The direct path reads a reply token's missing experts from storage
//! straight into their tables; the copying path loads them as a page and
//! copies it in afterwards. Both hand the multiply the same bytes, so a
//! seeded reply must come out token for token the same. This serves the
//! repository's long Shakespeare request three times on one loaded model,
//! under the configured MLX limit: copying, direct, and copying again. The
//! third run shows whether the copying path repeats itself, which is what a
//! difference between the first two would have to be judged against.
//!
//! Run through `cargo verify-direct-expert-read-parity` with a configuration
//! whose `model_directory` names the sparse model and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit.

use model_serving::Qwen3_5StreamingEngineTestControls;
use std::time::Duration;

use ipc_protocol::RequestId;
use runtime_integration::MlxMemoryLimits;
use serial_test::serial;
use test_harness::test_run::TestRunContext;
use tokio::time::timeout;

use super::decode_residency_attribution::{MEASURED_REQUEST_ID, probe_requests};
use crate::serving_acceptance::support::performance_attribution::{
    create_attributed_engine, load_engine_with_progress, run_attributed_generation,
};

const PARITY_TIMEOUT: Duration = Duration::from_secs(120);
/// Request ids of the three servings, apart from every other probe's.
const FIRST_PARITY_REQUEST_ID: u64 = 97_300;

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare request three times on an installed sparse model and compares the replies"]
#[serial]
async fn should_write_the_same_reply_whether_experts_are_read_into_tables_or_copied() {
    timeout(PARITY_TIMEOUT, compare_replies())
        .await
        .expect("the direct expert read parity probe must finish within 120 seconds");
}

async fn compare_replies() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_run_configuration = TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model to compare");
    let mlx_limit_bytes = usize::try_from(
        test_run_configuration
            .paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    let attribution_directory =
        tempfile::tempdir().expect("the probe should create an attribution directory");
    let mlx_memory_limits = MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
        .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_directory.path().join("parity.jsonl"),
        &mlx_memory_limits,
        test_run_configuration
            .paging_experiment
            .streaming_prompt_processing_chunk_tokens,
    );
    load_engine_with_progress(&mut engine, "direct_expert_read_parity_model_load").await;
    let mut measured_request = None;
    for probe_request in probe_requests(&model_directory) {
        if probe_request.request_id == MEASURED_REQUEST_ID {
            measured_request = Some(probe_request);
        }
    }
    let measured_request = measured_request.expect("the probe requests hold the long request");
    let mut replies = Vec::new();
    for (serving_index, (reads_into_tables, phase_name)) in [
        (false, "parity_copying"),
        (true, "parity_direct"),
        (false, "parity_copying_again"),
    ]
    .into_iter()
    .enumerate()
    {
        engine
            .set_decode_misses_fetched_into_tables_for_tests(reads_into_tables)
            .await
            .expect("the engine takes the read path");
        let reply_token_ids = run_attributed_generation(
            &mut engine,
            RequestId::new(FIRST_PARITY_REQUEST_ID + serving_index as u64),
            &measured_request.prompt_token_ids,
            phase_name,
            measured_request.output_token_count,
            &end_of_sequence_token_ids,
        )
        .await;
        eprintln!(
            "[direct-expert-read-parity] {phase_name}: {} reply tokens, fingerprint {:016x}",
            reply_token_ids.len(),
            reply_fingerprint(&reply_token_ids)
        );
        replies.push(reply_token_ids);
    }
    let copying_repeats_itself = replies[0] == replies[2];
    eprintln!(
        "[direct-expert-read-parity] copying then copying again: {}",
        if copying_repeats_itself {
            "same reply"
        } else {
            "DIFFERENT replies"
        }
    );
    assert!(
        !replies[0].is_empty(),
        "the reply must hold tokens to compare"
    );
    assert_eq!(
        replies[1],
        replies[0],
        "the reply read into tables must be the copied reply, token for token (first difference at token {:?})",
        first_difference(&replies[0], &replies[1])
    );
}

/// A reply's tokens folded into one number (FNV-1a over each token's bytes),
/// printed so replies from two runs can be compared: the same request served
/// from an Atlas and from its stock source must print the same fingerprint.
fn reply_fingerprint(reply_token_ids: &[u32]) -> u64 {
    let mut fingerprint: u64 = 0xcbf2_9ce4_8422_2325;
    for token_id in reply_token_ids {
        for token_byte in token_id.to_le_bytes() {
            fingerprint ^= u64::from(token_byte);
            fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fingerprint
}

/// Index of the first token two replies disagree on, or the shorter one's
/// length when one is a prefix of the other; `None` when they are equal.
fn first_difference(first_reply: &[u32], second_reply: &[u32]) -> Option<usize> {
    for (token_index, (first_token, second_token)) in
        first_reply.iter().zip(second_reply).enumerate()
    {
        if first_token != second_token {
            return Some(token_index);
        }
    }
    if first_reply.len() == second_reply.len() {
        None
    } else {
        Some(first_reply.len().min(second_reply.len()))
    }
}
