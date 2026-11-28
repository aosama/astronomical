//! A reply forward refused room for an expert page must run again and give
//! the same answer (issue #2749).
//!
//! On a Mac with little memory to spare a reply could end because one
//! streamed page was a few megabytes short, though the experts kept in
//! memory could have made that room many times over. The refusal depends on
//! what else the Mac is doing, so this journey has a test control refuse the
//! next page a reply streams, once, worded as admission words it. The same
//! question is asked before and after: the engine must count a redo and
//! write the reply to its end, and with the drafter silent the answer must
//! not change, which also shows the decoder state went back to where the
//! refused forward began.

use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::{Qwen3_5TextEngine, Qwen3_5TextEngineTestControls};
use serial_test::serial;
use tokio::time::timeout;

use super::paging_answer_parity::{PagingDepth, paging_ceiling_limits, parity_model_directory};
use crate::serving_acceptance::support::performance_attribution::{
    create_attributed_engine, load_engine_with_progress, run_attributed_generation,
};

const PROMPT_TOKEN_COUNT: usize = 1_024;
/// One chunk holds the prompt whole.
const PREFILL_CHUNK_TOKEN_COUNT: u32 = 1_024;
const OUTPUT_TOKEN_COUNT: u16 = 64;
const JOURNEY_BUDGET: Duration = Duration::from_secs(120);
const UNDISTURBED_REQUEST_ID: RequestId = RequestId::new(96_210);
const REFUSED_PAGE_REQUEST_ID: RequestId = RequestId::new(96_211);

/// One answer to the journey's question, drawn with the fixed seed the
/// journeys share: a seeded draw flips on logit changes far smaller than a
/// greedy pick ever sees.
async fn ask(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    request_id: RequestId,
    (prompt_token_ids, end_of_sequence_token_ids): (&[u32], &[u32]),
) -> Vec<u32> {
    let phase_name = "refused_reply_page";
    run_attributed_generation(
        qwen3_5_engine,
        request_id,
        prompt_token_ids,
        phase_name,
        OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
    )
    .await
}

async fn redo_attempt_count(qwen3_5_engine: &Qwen3_5TextEngine) -> u64 {
    let (_, redo_attempt_count) = qwen3_5_engine
        .memory_redo_counts_for_tests()
        .await
        .expect("the loaded model should count its redos");
    redo_attempt_count
}

/// With `drafter_writes_replies` the refused page is one a drafted forward of
/// several rows streams, the forward a small Mac's replies are made of.
async fn prove_a_refused_reply_page_is_run_again(drafter_writes_replies: bool) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = parity_model_directory();
    let attribution_directory =
        tempfile::tempdir().expect("the journey should create an attribution directory");
    let (mut qwen3_5_engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_directory
            .path()
            .join("refused-reply-page.jsonl"),
        // Half the experts are out of RAM, so a reply streams pages.
        &paging_ceiling_limits(&model_directory, PagingDepth::HalfOfExperts),
        PREFILL_CHUNK_TOKEN_COUNT,
    );
    load_engine_with_progress(&mut qwen3_5_engine, "refused_reply_page").await;
    qwen3_5_engine
        .set_hosted_drafter_writes_replies_for_tests(drafter_writes_replies)
        .await
        .expect("the engine takes the drafter's part");
    let prompt_token_ids = crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &model_directory,
        crate::common::resident_sparse_moe_model_id(),
        UNDISTURBED_REQUEST_ID,
        PROMPT_TOKEN_COUNT,
        OUTPUT_TOKEN_COUNT,
    );
    let undisturbed_token_ids = ask(
        &mut qwen3_5_engine,
        UNDISTURBED_REQUEST_ID,
        (&prompt_token_ids, &end_of_sequence_token_ids),
    )
    .await;
    let redo_attempt_count_before = redo_attempt_count(&qwen3_5_engine).await;
    qwen3_5_engine
        .refuse_next_reply_page_once_for_tests()
        .await
        .expect("the engine should accept the page to refuse");
    let refused_page_token_ids = ask(
        &mut qwen3_5_engine,
        REFUSED_PAGE_REQUEST_ID,
        (&prompt_token_ids, &end_of_sequence_token_ids),
    )
    .await;
    let redone_forward_count =
        redo_attempt_count(&qwen3_5_engine).await - redo_attempt_count_before;
    eprintln!(
        "[refused-reply-page] status=success drafter_writes_replies={drafter_writes_replies} redone_forwards={redone_forward_count} reply_tokens={}",
        refused_page_token_ids.len()
    );
    assert!(
        redone_forward_count >= 1,
        "the reply that had a page refused must have run a forward again"
    );
    assert_eq!(
        undisturbed_token_ids.len(),
        refused_page_token_ids.len(),
        "the reply that had a page refused must be written to its end"
    );
    // A drafted reply depends on which drafts were kept, and the batched
    // check that keeps them differs from the one-row step in the last bits,
    // so two drafted asks differ with no refusal at all. Only the plain
    // reply is compared token for token.
    if !drafter_writes_replies {
        assert_eq!(
            undisturbed_token_ids, refused_page_token_ids,
            "running a refused reply forward again changed the answer"
        );
    }
}

/// Runs the journey inside its budget, so a reply that never ends fails here.
async fn prove_within_the_journey_budget(drafter_writes_replies: bool) {
    timeout(
        JOURNEY_BUDGET,
        prove_a_refused_reply_page_is_run_again(drafter_writes_replies),
    )
    .await
    .expect("the refused reply page journey should finish within its budget");
}

#[tokio::test]
#[ignore = "loads a sparse model with half its experts out of RAM"]
#[serial]
async fn should_give_the_same_reply_after_a_reply_page_is_refused_room() {
    prove_within_the_journey_budget(false).await;
}

#[tokio::test]
#[ignore = "loads the same model and has its drafter write the reply"]
#[serial]
async fn should_finish_a_drafted_reply_after_a_reply_page_is_refused_room() {
    prove_within_the_journey_budget(true).await;
}
