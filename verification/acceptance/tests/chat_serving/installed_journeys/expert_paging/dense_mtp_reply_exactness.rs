//! A dense pack's multi-token-prediction head (Ternary Bonsai 2 9B's `mtp/`)
//! drafts replies that are the model's own.
//!
//! One loaded model writes one greedy reply three times: one token a
//! forward, then with one and with three drafted tokens a round. Every kept
//! token is the model's own pick after a verified row, so a drafted reply is
//! the plain one up to float16 rounding: a several-row verify sums in another
//! order than a one-row step, and a near-tie can then fall the other way.
//! The journey fails when a drafted reply did not draft (the head was not
//! hosted), when the head's drafts were kept too rarely to be the head's own
//! work, or when the replies part within their first sixteen tokens, which
//! rounding does not explain. It prints each drafted reply's stages from the
//! engine's own tally: rounds, rows verified, and the time drafting,
//! verifying, picking and keeping took, attribution off.
//!
//! Run with `cargo test-dense-mtp-reply --configuration PATH`, whose
//! `model_directory` names a dense pack with an `mtp/` head. About a minute.

use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::Qwen3_5TextEngineTestControls;
use serial_test::serial;
use tokio::time::timeout;

use super::reply_speed_cases::write_compared_greedy_reply;
use crate::serving_acceptance::support::performance_attribution::{
    PagingExperimentEngine, load_paging_experiment_engine,
};

const PLAIN_REQUEST_ID: RequestId = RequestId::new(98_701);
const ONE_DRAFT_REQUEST_ID: RequestId = RequestId::new(98_702);
const THREE_DRAFTS_REQUEST_ID: RequestId = RequestId::new(98_703);
const PROOF_TIMEOUT: Duration = Duration::from_secs(300);
/// Replies that part earlier than this disagree for a reason other than a
/// float16 near-tie: Ternary Bonsai 2 9B's agreed on all 96 tokens.
const MINIMUM_AGREEING_TOKEN_COUNT: usize = 16;
/// Tokens a round must average with one draft for the drafts to be the
/// head's own work: the 9B's head kept 56% of its drafts at temperature 1.0,
/// 1.56 tokens a round, and a head guessing at random keeps almost none.
const MINIMUM_TOKENS_A_ONE_DRAFT_ROUND: f64 = 1.25;

#[tokio::test]
#[ignore = "writes one greedy reply plain and drafted on an installed dense pack with an mtp head"]
#[serial]
async fn should_write_the_plain_greedy_reply_when_a_dense_head_drafts() {
    timeout(PROOF_TIMEOUT, compare_drafted_with_plain_replies())
        .await
        .expect("the dense head proof must finish within its budget");
}

async fn compare_drafted_with_plain_replies() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut paging_engine = load_paging_experiment_engine("dense_mtp_model_load").await;
    let plain_reply = write_reply_drafting(&mut paging_engine, 0, PLAIN_REQUEST_ID).await;
    for (draft_token_count, request_id) in [(1, ONE_DRAFT_REQUEST_ID), (3, THREE_DRAFTS_REQUEST_ID)]
    {
        let drafted_reply =
            write_reply_drafting(&mut paging_engine, draft_token_count, request_id).await;
        let stage_counts = last_reply_stage_counts(&paging_engine).await;
        let agreeing_token_count = agreeing_prefix_length(&drafted_reply, &plain_reply);
        report_drafted_reply(
            draft_token_count,
            &drafted_reply,
            &plain_reply,
            &stage_counts,
        );
        assert!(
            agreeing_token_count >= MINIMUM_AGREEING_TOKEN_COUNT.min(plain_reply.len()),
            "a drafted reply must agree with the plain reply past rounding: {agreeing_token_count}"
        );
        assert_the_head_drafted(draft_token_count, drafted_reply.len(), &stage_counts);
    }
}

/// What the engine's own tally says about a drafted reply's rounds.
struct DraftedReplyStages {
    draft_round_count: u64,
    reply_row_count: u64,
    draft_nanoseconds: u64,
    verify_nanoseconds: u64,
    pick_nanoseconds: u64,
    keep_nanoseconds: u64,
}

async fn last_reply_stage_counts(paging_engine: &PagingExperimentEngine) -> DraftedReplyStages {
    let stage_counts = paging_engine
        .engine
        .last_reply_stage_counts_for_tests()
        .await
        .expect("the engine reports its reply stages")
        .expect("a drafted reply leaves its stage counts");
    DraftedReplyStages {
        draft_round_count: stage_counts.draft_round_count,
        reply_row_count: stage_counts.reply_row_count,
        draft_nanoseconds: stage_counts.draft_nanoseconds,
        verify_nanoseconds: stage_counts.verify_nanoseconds,
        pick_nanoseconds: stage_counts.pick_nanoseconds,
        keep_nanoseconds: stage_counts.keep_nanoseconds,
    }
}

/// The reply drafted, and with one draft a round the head's drafts were
/// kept often enough to be its own work rather than chance.
fn assert_the_head_drafted(
    draft_token_count: u32,
    reply_token_count: usize,
    stage_counts: &DraftedReplyStages,
) {
    assert!(
        stage_counts.draft_round_count > 0,
        "the reply drafted no round: the pack's head was not hosted"
    );
    if draft_token_count != 1 {
        return;
    }
    let tokens_a_round = reply_token_count as f64 / stage_counts.draft_round_count as f64;
    assert!(
        tokens_a_round >= MINIMUM_TOKENS_A_ONE_DRAFT_ROUND,
        "one-draft rounds averaged {tokens_a_round:.2} tokens: the head's drafts are rarely kept"
    );
}

/// One line a drafted reply: agreement with the plain reply, rounds, rows,
/// and each stage's mean time a round in microseconds.
fn report_drafted_reply(
    draft_token_count: u32,
    drafted_reply: &[u32],
    plain_reply: &[u32],
    stage_counts: &DraftedReplyStages,
) {
    let rounds = stage_counts.draft_round_count.max(1);
    eprintln!(
        "[dense-mtp-proof] draft_tokens={draft_token_count} agreeing_tokens={} of {} identical={} rounds={} verified_rows={} draft_us={} verify_us={} pick_us={} keep_us={}",
        agreeing_prefix_length(drafted_reply, plain_reply),
        plain_reply.len(),
        drafted_reply == plain_reply,
        stage_counts.draft_round_count,
        stage_counts.reply_row_count,
        stage_counts.draft_nanoseconds / rounds / 1_000,
        stage_counts.verify_nanoseconds / rounds / 1_000,
        stage_counts.pick_nanoseconds / rounds / 1_000,
        stage_counts.keep_nanoseconds / rounds / 1_000,
    );
}

async fn write_reply_drafting(
    paging_engine: &mut PagingExperimentEngine,
    draft_token_count: u32,
    request_id: RequestId,
) -> Vec<u32> {
    paging_engine
        .engine
        .set_multi_token_prediction_draft_tokens_for_tests(draft_token_count)
        .await
        .expect("the engine takes the draft count");
    write_compared_greedy_reply(
        paging_engine,
        request_id,
        &format!("dense_mtp_drafts_{draft_token_count}"),
    )
    .await
}

/// How many leading tokens the two replies share.
fn agreeing_prefix_length(drafted_reply: &[u32], plain_reply: &[u32]) -> usize {
    let mut agreeing_token_count = 0;
    while agreeing_token_count < drafted_reply.len().min(plain_reply.len())
        && drafted_reply[agreeing_token_count] == plain_reply[agreeing_token_count]
    {
        agreeing_token_count += 1;
    }
    agreeing_token_count
}
