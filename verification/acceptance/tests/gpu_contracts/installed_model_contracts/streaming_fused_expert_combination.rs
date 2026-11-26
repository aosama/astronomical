//! The engine joins a prompt chunk's routed experts and its
//! shared expert in one fused Metal pass (issue #2585), as the in-memory
//! engine does. The kernel is proven bit for bit against the two steps it
//! replaces on synthetic rows; this proves the same on a served model's own
//! layers: one load of a streamed mixture-of-experts model answers the
//! Shakespeare prompt with the two-step combination and with the fused one.
//!
//! Run with `cargo test-streaming-fused-expert-combination`; append
//! `--configuration PATH` naming a `model_directory` to prove it on another
//! mixture-of-experts model.

use std::time::Duration;

use model_serving::Qwen3_5TextModel;
use serial_test::serial;
use tokio::time::timeout;

use super::greedy_answer::{
    GreedyAnswer, greedy_answer, load_materialized_streaming_model, shakespeare_prompt_token_ids,
};

const COMBINATION_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads a streamed mixture-of-experts model and answers a prompt twice"]
#[serial]
async fn should_give_the_same_reply_and_logits_with_the_fused_expert_combination() {
    timeout(
        COMBINATION_BUDGET,
        assert_fused_combination_changes_nothing(),
    )
    .await
    .expect("the fused combination journey must finish within its budget");
}

async fn assert_fused_combination_changes_nothing() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = mixture_of_experts_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    let (mut streaming_model, qwen3_5_config) =
        load_materialized_streaming_model(&model_directory).await;

    // Positive: a GPU that passes the capability probe keeps the kernel at
    // load, with nothing asked of the person.
    assert!(
        fused_kernel_is_kept(&streaming_model),
        "the engine should keep the fused expert combination at load"
    );
    let fused_answer = greedy_answer(&streaming_model, &qwen3_5_config, &prompt_token_ids);

    // Negative: without the kernel the two-step path serves, and it is the
    // reference the fused pass must equal.
    streaming_model.disable_fused_sorted_expert_shared_combination();
    assert!(!fused_kernel_is_kept(&streaming_model));
    let two_step_answer = greedy_answer(&streaming_model, &qwen3_5_config, &prompt_token_ids);
    assert_eq!(
        fused_answer.generated_token_ids, two_step_answer.generated_token_ids,
        "the fused combination must write the reply the two-step one writes"
    );
    assert_eq!(
        logit_bits(&fused_answer),
        logit_bits(&two_step_answer),
        "the fused combination must leave every logit's bits unchanged"
    );

    // Edge: switching it back on gives the first answer again, so the
    // switch itself leaves no state behind.
    streaming_model
        .enable_fused_sorted_expert_shared_combination()
        .expect("the fused kernel should build");
    let fused_again_answer = greedy_answer(&streaming_model, &qwen3_5_config, &prompt_token_ids);
    assert_eq!(logit_bits(&fused_again_answer), logit_bits(&fused_answer));
    report_alternated_answer_times(&mut streaming_model, &qwen3_5_config, &prompt_token_ids);
    eprintln!(
        "[streaming-fused-expert-combination] status=success prompt_tokens={} reply_tokens={} logits={}",
        prompt_token_ids.len(),
        fused_answer.generated_token_ids.len(),
        fused_answer.final_logits.len()
    );
}

/// Pairs of answers timed with the order flipped every pair.
const TIMED_PAIR_COUNT: usize = 4;

/// Prints the median seconds of an answer (the prompt read in chunks, then a
/// few tokens) each way, alternated in one process so drift lands on both.
/// It asserts nothing: the kernel's gain is a fraction of a percent.
fn report_alternated_answer_times(
    streaming_model: &mut Qwen3_5TextModel,
    qwen3_5_config: &model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
) {
    let (mut two_step_seconds, mut fused_seconds) = (Vec::new(), Vec::new());
    for pair_index in 0..TIMED_PAIR_COUNT {
        for is_fused in [pair_index % 2 == 0, pair_index % 2 != 0] {
            if is_fused {
                streaming_model
                    .enable_fused_sorted_expert_shared_combination()
                    .expect("the fused kernel should build");
            } else {
                streaming_model.disable_fused_sorted_expert_shared_combination();
            }
            let started_at = std::time::Instant::now();
            greedy_answer(streaming_model, qwen3_5_config, prompt_token_ids);
            let answer_seconds = started_at.elapsed().as_secs_f64();
            if is_fused {
                fused_seconds.push(answer_seconds);
            } else {
                two_step_seconds.push(answer_seconds);
            }
        }
    }
    // Leave the model as it was loaded.
    streaming_model
        .enable_fused_sorted_expert_shared_combination()
        .expect("the fused kernel should build");
    two_step_seconds.sort_by(f64::total_cmp);
    fused_seconds.sort_by(f64::total_cmp);
    eprintln!(
        "[streaming-fused-expert-combination] answer_seconds two_step median={:.4} range={:.4}..{:.4} | fused median={:.4} range={:.4}..{:.4}",
        two_step_seconds[TIMED_PAIR_COUNT / 2],
        two_step_seconds[0],
        two_step_seconds[TIMED_PAIR_COUNT - 1],
        fused_seconds[TIMED_PAIR_COUNT / 2],
        fused_seconds[0],
        fused_seconds[TIMED_PAIR_COUNT - 1],
    );
}

/// The run's configured model when it names one, else the small streamed
/// mixture-of-experts model the acceptance roles name.
fn mixture_of_experts_model_directory() -> std::path::PathBuf {
    test_harness::test_run::TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(crate::common::configured_small_sparse_moe_streaming_model_directory)
}

fn fused_kernel_is_kept(streaming_model: &Qwen3_5TextModel) -> bool {
    streaming_model
        .optional_accelerators
        .sorted_expert_shared_combination_kernel
        .is_some()
}

fn logit_bits(greedy_answer: &GreedyAnswer) -> Vec<u32> {
    let mut bits = Vec::with_capacity(greedy_answer.final_logits.len());
    for logit in &greedy_answer.final_logits {
        bits.push(logit.to_bits());
    }
    bits
}
