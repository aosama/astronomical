//! Qwen3.8-Flash-Next's speed at the lowest layer: the streaming model's own
//! forwards, with each reply token sampled at the temperatures people use
//! (0.7 for code, 1.0 for everything else), at a 34 GiB limit (issue #2533).
//!
//! The model is put into the memory state the engine serves in (the
//! governor open, the `astro_expert_cache` sized for a prompt and then for a
//! reply), so the expert reads are the ones a served reply pays. A 500-token
//! warmup with 50 reply tokens runs first; then a 10,000-token Romeo and
//! Juliet prompt is read in 4,096-token chunks and 500 reply tokens are
//! sampled one forward each. The temperature changes which tokens are
//! chosen, and so which experts the next forward routes to; the forward
//! itself is the same. Progress and time left print every five seconds.
//!
//! Run with `cargo measure-flash-next-sampled-forward-speed`. About ninety
//! seconds a temperature.

use std::time::{Duration, Instant};

use model_serving::qwen3_5_streaming::model::mtp_drafting::{TargetTokenChoice, TopKTopPSampling};
use model_serving::{MemoryPhase, Qwen3_5StreamingModel};
use serial_test::file_serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// The engine's SSD-streaming prompt chunk.
const PROMPT_CHUNK_TOKEN_COUNT: usize = 4_096;
const WARMUP_PROMPT_WORD_COUNT: usize = 320;
const WARMUP_REPLY_TOKEN_COUNT: usize = 50;
/// About 10,000 tokens of Romeo and Juliet with this tokenizer.
const MEASURED_PROMPT_WORD_COUNT: usize = 6_496;
const MEASURED_REPLY_TOKEN_COUNT: usize = 500;
/// The app's sampling defaults: top-k 20, no top-p cut.
const TOP_K: u16 = 20;
const TOP_P_THOUSANDTHS: u16 = 1_000;
const KEYED_SEED: u64 = 1_234;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const CASE_BUDGET: Duration = Duration::from_secs(150);

#[tokio::test]
#[ignore = "loads Qwen3.8-Flash-Next and samples a long reply at temperature 0.7"]
#[file_serial]
async fn should_measure_flash_next_forward_speed_at_temperature_0_7() {
    timeout(CASE_BUDGET, measure_sampled_forward_speed(700))
        .await
        .expect("the sampled forward speed case must finish within its budget");
}

#[tokio::test]
#[ignore = "loads Qwen3.8-Flash-Next and samples a long reply at temperature 1.0"]
#[file_serial]
async fn should_measure_flash_next_forward_speed_at_temperature_1_0() {
    timeout(CASE_BUDGET, measure_sampled_forward_speed(1_000))
        .await
        .expect("the sampled forward speed case must finish within its budget");
}

async fn measure_sampled_forward_speed(temperature_thousandths: u16) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::flash_next_loading::flash_next_model_directory();
    let streaming_model = super::flash_next_loading::load_flash_next(&model_directory);
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the tokenizer should load");
    let token_choice = TargetTokenChoice {
        sampling: Some(TopKTopPSampling {
            temperature_thousandths,
            top_k: TOP_K,
            top_p_thousandths: TOP_P_THOUSANDTHS,
        }),
        keyed_seed: KEYED_SEED,
    };
    sampled_reply(
        &streaming_model,
        &shakespeare_prompt_token_ids(&tokenizer, WARMUP_PROMPT_WORD_COUNT),
        WARMUP_REPLY_TOKEN_COUNT,
        token_choice,
        "warmup",
    );
    let measured = sampled_reply(
        &streaming_model,
        &shakespeare_prompt_token_ids(&tokenizer, MEASURED_PROMPT_WORD_COUNT),
        MEASURED_REPLY_TOKEN_COUNT,
        token_choice,
        "measured",
    );
    eprintln!(
        "[flash-next-forward-speed] temperature_thousandths={temperature_thousandths} prompt_tokens={} prefill_tokens_per_second={:.1} reply_tokens={MEASURED_REPLY_TOKEN_COUNT} decode_tokens_per_second={:.2} peak_mlx_gigabytes={:.2}",
        measured.prompt_token_count,
        measured.prefill_tokens_per_second,
        measured.decode_tokens_per_second,
        streaming_model
            .runtime()
            .memory_snapshot()
            .expect("the memory snapshot reads")
            .peak_memory_bytes() as f64
            / 1_073_741_824.0,
    );
    assert!(
        measured.decode_tokens_per_second > 0.0,
        "the sampled reply must produce tokens"
    );
}

fn shakespeare_prompt_token_ids(tokenizer: &tokenizers::Tokenizer, word_count: usize) -> Vec<u32> {
    tokenizer
        .encode(romeo_and_juliet::summary_prompt(word_count), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec()
}

/// One reply's speeds and size.
struct SampledReplySpeed {
    prompt_token_count: usize,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
}

/// Reads the prompt in engine-sized chunks and samples `reply_token_count`
/// tokens, each from one forward, in the memory state the engine serves in.
fn sampled_reply(
    streaming_model: &Qwen3_5StreamingModel,
    prompt_token_ids: &[u32],
    reply_token_count: usize,
    token_choice: TargetTokenChoice,
    label: &str,
) -> SampledReplySpeed {
    let mut attribution = model_serving::PerformanceAttribution::disabled();
    streaming_model
        .enter_memory_phase_for_tests(
            MemoryPhase::Prefill,
            0,
            PROMPT_CHUNK_TOKEN_COUNT as u64,
            &mut attribution,
        )
        .expect("the model enters the prompt phase");
    let mut request_decoder_state =
        crate::common::standard_request_decoder_state(&streaming_model.config);
    let prefill_started = Instant::now();
    let mut last_logits = None;
    for chunk_start in (0..prompt_token_ids.len()).step_by(PROMPT_CHUNK_TOKEN_COUNT) {
        last_logits = Some(super::flash_next_loading::forward_prompt_chunk(
            streaming_model,
            prompt_token_ids,
            chunk_start,
            PROMPT_CHUNK_TOKEN_COUNT,
            &mut request_decoder_state,
        ));
        let chunk_end = (chunk_start + PROMPT_CHUNK_TOKEN_COUNT).min(prompt_token_ids.len());
        report_progress(
            label,
            "prompt",
            chunk_end,
            prompt_token_ids.len(),
            prefill_started,
        );
    }
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    streaming_model
        .enter_memory_phase_for_tests(
            MemoryPhase::Decode,
            prompt_token_ids.len() as u64,
            1,
            &mut attribution,
        )
        .expect("the model enters the reply phase");
    let decode_started = Instant::now();
    let mut last_report = Instant::now();
    let mut next_logits = last_logits.expect("the prompt has tokens");
    for reply_index in 0..reply_token_count {
        let position = (prompt_token_ids.len() + reply_index) as u32;
        let chosen_token_id = streaming_model
            .target_token_picks(&next_logits, position.saturating_sub(1), token_choice)
            .expect("the reply token is sampled")[0];
        next_logits = streaming_model
            .forward_chunk(&[chosen_token_id], position, &mut request_decoder_state)
            .expect("the reply token should forward");
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            report_progress(
                label,
                "reply",
                reply_index + 1,
                reply_token_count,
                decode_started,
            );
        }
    }
    SampledReplySpeed {
        prompt_token_count: prompt_token_ids.len(),
        prefill_tokens_per_second: prompt_token_ids.len() as f64 / prefill_seconds,
        decode_tokens_per_second: reply_token_count as f64 / decode_started.elapsed().as_secs_f64(),
    }
}

fn report_progress(label: &str, stage: &str, completed: usize, total: usize, started: Instant) {
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let seconds_left = elapsed_seconds / completed.max(1) as f64 * (total - completed) as f64;
    eprintln!(
        "[flash-next-forward-speed] {label} {stage} {completed}/{total} ({:.0}%) elapsed {elapsed_seconds:.1}s, about {seconds_left:.0}s left",
        100.0 * completed as f64 / total as f64
    );
}
