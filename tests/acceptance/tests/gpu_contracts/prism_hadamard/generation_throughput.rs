//! Tier 4: greedy generation through the model's own session.
//!
//! The session owns prompt processing (2,048-token chunks) and the decode
//! loop; the test times the two phases separately. A warmup session of
//! roughly 500 prompt tokens and 50 generated tokens precedes the measured
//! session of at least 10,000 prompt tokens and 500 generated tokens. No stop
//! ids are supplied, so the measured output count is the budget itself.

use std::time::Instant;

use astronomical_model_serving::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use serial_test::serial;

use super::support::{
    MEASURED_INPUT_TOKENS, MEASURED_OUTPUT_TOKENS, PREFILL_CHUNK_TOKENS, WARMUP_INPUT_TOKENS,
    WARMUP_OUTPUT_TOKENS, assert_measured_shape, loaded_journey_model, pack_tokenizer,
    romeo_and_juliet_prompt_token_ids,
};

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and measures greedy generation throughput on the graphics processor"]
fn should_measure_greedy_generation_throughput() {
    let model = loaded_journey_model();
    let tokenizer = pack_tokenizer();

    let warmup_prompt_token_ids =
        romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS);
    let warmup = run_generation_journey(&model, &warmup_prompt_token_ids, WARMUP_OUTPUT_TOKENS);
    eprintln!(
        "[prism-hadamard-perf:tier=generation] phase=warmup prompt_tokens={} generated_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2}",
        warmup_prompt_token_ids.len(),
        warmup.generated_token_count,
        warmup.prefill_tokens_per_second,
        warmup.decode_tokens_per_second,
    );

    let measured_prompt_token_ids =
        romeo_and_juliet_prompt_token_ids(&tokenizer, MEASURED_INPUT_TOKENS);
    let measured =
        run_generation_journey(&model, &measured_prompt_token_ids, MEASURED_OUTPUT_TOKENS);
    eprintln!(
        "[prism-hadamard-perf:tier=generation] phase=measured prompt_tokens={} prefill_chunk_tokens={PREFILL_CHUNK_TOKENS} prefill_seconds={:.2} prefill_tokens_per_second={:.1} generated_tokens={} decode_seconds={:.2} decode_tokens_per_second={:.2}",
        measured_prompt_token_ids.len(),
        measured.prefill_seconds,
        measured.prefill_tokens_per_second,
        measured.generated_token_count,
        measured.decode_seconds,
        measured.decode_tokens_per_second,
    );
    assert_measured_shape(
        measured_prompt_token_ids.len(),
        measured.generated_token_count,
    );
    assert!(
        measured.prefill_tokens_per_second.is_finite() && measured.prefill_tokens_per_second > 0.0
    );
    assert!(
        measured.decode_tokens_per_second.is_finite() && measured.decode_tokens_per_second > 0.0
    );
}

struct GenerationJourneyMeasurement {
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    generated_token_count: usize,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
}

fn run_generation_journey(
    model: &PrismHadamardTextModel,
    prompt_token_ids: &[i32],
    generated_token_budget: usize,
) -> GenerationJourneyMeasurement {
    let prefill_started_at = Instant::now();
    let mut session = model
        .begin_greedy_session(
            prompt_token_ids,
            PREFILL_CHUNK_TOKENS as u32,
            generated_token_budget,
            &[],
        )
        .expect("the greedy session should process the prompt");
    let prefill_seconds = prefill_started_at.elapsed().as_secs_f64();

    let decode_started_at = Instant::now();
    let mut generated_token_count = 0;
    while let Some(_token_id) = session
        .next_token()
        .expect("the greedy session should decode")
    {
        generated_token_count += 1;
    }
    let decode_seconds = decode_started_at.elapsed().as_secs_f64();
    GenerationJourneyMeasurement {
        prefill_seconds,
        prefill_tokens_per_second: prompt_token_ids.len() as f64 / prefill_seconds,
        generated_token_count,
        decode_seconds,
        decode_tokens_per_second: generated_token_count as f64 / decode_seconds,
    }
}
