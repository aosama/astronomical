//! Tier 3: the whole-model forward pass, prefill chunks and decode steps.
//!
//! After a warmup of roughly 500 prompt tokens and 50 decode steps, a Romeo
//! and Juliet prompt of at least 10,000 tokens is processed in 2,048-token
//! chunks (each chunk evaluated before the next is built, as serving does),
//! then 500 greedy decode steps run through the raw forward pass with one
//! host synchronization per step. This tier isolates the forward graph from
//! the generation session's scheduling, so the gap to tier 4 is the
//! scheduling gain alone.

use std::time::Instant;

use astronomical_model_serving::prism_hadamard_qwen35::model::{
    PrismHadamardGenerationState, PrismHadamardTextModel,
};
use serial_test::serial;

use super::support::{
    MEASURED_INPUT_TOKENS, MEASURED_OUTPUT_TOKENS, PREFILL_CHUNK_TOKENS, WARMUP_INPUT_TOKENS,
    WARMUP_OUTPUT_TOKENS, assert_measured_shape, loaded_journey_model, median, pack_tokenizer,
    romeo_and_juliet_prompt_token_ids,
};

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and measures raw prefill-chunk and decode-step throughput on the graphics processor"]
fn should_measure_forward_pass_prefill_and_decode_throughput() {
    let model = loaded_journey_model();
    let tokenizer = pack_tokenizer();

    let warmup_prompt_token_ids =
        romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS);
    let warmup = run_forward_journey(&model, &warmup_prompt_token_ids, WARMUP_OUTPUT_TOKENS);
    eprintln!(
        "[prism-hadamard-perf:tier=forward] phase=warmup prompt_tokens={} generated_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2}",
        warmup_prompt_token_ids.len(),
        warmup.generated_token_count,
        warmup.prefill_tokens_per_second,
        warmup.decode_tokens_per_second,
    );

    let measured_prompt_token_ids =
        romeo_and_juliet_prompt_token_ids(&tokenizer, MEASURED_INPUT_TOKENS);
    let measured = run_forward_journey(&model, &measured_prompt_token_ids, MEASURED_OUTPUT_TOKENS);
    eprintln!(
        "[prism-hadamard-perf:tier=forward] phase=measured prompt_tokens={} prefill_chunk_tokens={PREFILL_CHUNK_TOKENS} prefill_seconds={:.2} prefill_tokens_per_second={:.1} generated_tokens={} decode_seconds={:.2} decode_tokens_per_second={:.2} decode_step_median_ms={:.2}",
        measured_prompt_token_ids.len(),
        measured.prefill_seconds,
        measured.prefill_tokens_per_second,
        measured.generated_token_count,
        measured.decode_seconds,
        measured.decode_tokens_per_second,
        measured.decode_step_median_millis,
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

struct ForwardJourneyMeasurement {
    prefill_seconds: f64,
    prefill_tokens_per_second: f64,
    generated_token_count: usize,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
    decode_step_median_millis: f64,
}

/// Prefills in fixed chunks, then decodes greedily through the raw forward
/// pass: one evaluation and one argmax synchronization per step.
fn run_forward_journey(
    model: &PrismHadamardTextModel,
    prompt_token_ids: &[i32],
    generated_token_budget: usize,
) -> ForwardJourneyMeasurement {
    let mut state = PrismHadamardGenerationState::empty();
    let prefill_started_at = Instant::now();
    let mut final_logits = None;
    for chunk in prompt_token_ids.chunks(PREFILL_CHUNK_TOKENS) {
        let chunk_logits = model
            .prefill_chunk(chunk, &mut state)
            .expect("the prefill chunk should build");
        model
            .evaluate_step(&chunk_logits, &state)
            .expect("the prefill chunk should evaluate");
        final_logits = Some(chunk_logits);
    }
    let prefill_seconds = prefill_started_at.elapsed().as_secs_f64();

    let mut logits = final_logits.expect("the prompt should produce final logits");
    let mut step_samples = Vec::with_capacity(generated_token_budget);
    let decode_started_at = Instant::now();
    let mut generated_token_count = 0;
    while generated_token_count < generated_token_budget {
        let step_started_at = Instant::now();
        let next_token_id = model
            .greedy_token_id(&logits)
            .expect("the greedy token should synchronize");
        generated_token_count += 1;
        logits = model
            .decode_step(next_token_id, &mut state)
            .expect("the decode step should build");
        model
            .evaluate_step(&logits, &state)
            .expect("the decode step should evaluate");
        step_samples.push(step_started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    let decode_seconds = decode_started_at.elapsed().as_secs_f64();
    ForwardJourneyMeasurement {
        prefill_seconds,
        prefill_tokens_per_second: prompt_token_ids.len() as f64 / prefill_seconds,
        generated_token_count,
        decode_seconds,
        decode_tokens_per_second: generated_token_count as f64 / decode_seconds,
        decode_step_median_millis: median(&mut step_samples),
    }
}
