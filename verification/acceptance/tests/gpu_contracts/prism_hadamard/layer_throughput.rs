//! Tier 2: one decoder layer of each attention kind.
//!
//! The first gated-delta layer and the first full-attention layer run over
//! real Romeo and Juliet embeddings, prefill-shaped (one 2,048-token chunk on
//! a fresh state) and decode-shaped (one row after that chunk populated the
//! layer's state, so attention and the recurrence see a realistic prefix).

use std::time::Instant;

use astronomical_model_serving::prism_hadamard_qwen35::model::{
    PrismHadamardGenerationState, PrismHadamardTextModel,
};
use serial_test::serial;

use super::support::{
    PREFILL_CHUNK_TOKENS, loaded_journey_model, materialized, median, pack_tokenizer,
    romeo_and_juliet_prompt_token_ids,
};

const PREFILL_WARMUP_ITERATIONS: usize = 2;
const PREFILL_MEASURED_ITERATIONS: usize = 5;
const DECODE_WARMUP_STEPS: usize = 10;
const DECODE_MEASURED_STEPS: usize = 50;

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and measures one gated-delta and one full-attention decoder layer on the graphics processor"]
fn should_measure_decoder_layer_throughput() {
    let model = loaded_journey_model();
    let tokenizer = pack_tokenizer();
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(
        &tokenizer,
        PREFILL_CHUNK_TOKENS + DECODE_WARMUP_STEPS + DECODE_MEASURED_STEPS,
    );
    let chunk_hidden_states = materialized(
        &model,
        model
            .probe_embedding(&prompt_token_ids[..PREFILL_CHUNK_TOKENS])
            .expect("the prefill chunk should embed"),
    );
    let (gated_delta_layer_index, full_attention_layer_index) = model.first_layer_indices_by_kind();
    let probed_layers = [
        ("gated_delta", gated_delta_layer_index),
        ("full_attention", full_attention_layer_index),
    ];
    for (layer_label, layer_index) in probed_layers {
        let Some(layer_index) = layer_index else {
            eprintln!("[prism-hadamard-perf:tier=layer] layer={layer_label} absent from the pack");
            continue;
        };
        report_prefill_shaped_layer(&model, layer_label, layer_index, &chunk_hidden_states);
        report_decode_shaped_layer(
            &model,
            layer_label,
            layer_index,
            &chunk_hidden_states,
            &prompt_token_ids[PREFILL_CHUNK_TOKENS..],
        );
    }
}

fn report_prefill_shaped_layer(
    model: &PrismHadamardTextModel,
    layer_label: &str,
    layer_index: usize,
    chunk_hidden_states: &mlx_c_rust::MlxArray,
) {
    let mut samples = Vec::with_capacity(PREFILL_MEASURED_ITERATIONS);
    for iteration in 0..PREFILL_WARMUP_ITERATIONS + PREFILL_MEASURED_ITERATIONS {
        let mut state = PrismHadamardGenerationState::empty();
        let started_at = Instant::now();
        let layer_output = model
            .probe_decoder_layer(layer_index, chunk_hidden_states, &mut state)
            .expect("the prefill-shaped layer should build");
        model
            .evaluate_step(&layer_output, &state)
            .expect("the prefill-shaped layer should evaluate");
        if iteration >= PREFILL_WARMUP_ITERATIONS {
            samples.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    let median_millis = median(&mut samples);
    eprintln!(
        "[prism-hadamard-perf:tier=layer] layer={layer_label} index={layer_index} shape=prefill tokens={PREFILL_CHUNK_TOKENS} median_ms={median_millis:.2} tokens_per_second={:.0}",
        PREFILL_CHUNK_TOKENS as f64 / (median_millis / 1_000.0),
    );
    assert!(median_millis.is_finite() && median_millis > 0.0);
}

fn report_decode_shaped_layer(
    model: &PrismHadamardTextModel,
    layer_label: &str,
    layer_index: usize,
    chunk_hidden_states: &mlx_c_rust::MlxArray,
    decode_token_ids: &[i32],
) {
    let mut state = PrismHadamardGenerationState::empty();
    let chunk_output = model
        .probe_decoder_layer(layer_index, chunk_hidden_states, &mut state)
        .expect("the state-populating chunk should build");
    model
        .evaluate_step(&chunk_output, &state)
        .expect("the state-populating chunk should evaluate");
    model.advance_probe_state(&mut state, PREFILL_CHUNK_TOKENS as i32);

    let mut samples = Vec::with_capacity(DECODE_MEASURED_STEPS);
    for (step, decode_token_id) in decode_token_ids
        .iter()
        .take(DECODE_WARMUP_STEPS + DECODE_MEASURED_STEPS)
        .enumerate()
    {
        let row_hidden_states = materialized(
            model,
            model
                .probe_embedding(&[*decode_token_id])
                .expect("the decode token should embed"),
        );
        let started_at = Instant::now();
        let layer_output = model
            .probe_decoder_layer(layer_index, &row_hidden_states, &mut state)
            .expect("the decode-shaped layer should build");
        model
            .evaluate_step(&layer_output, &state)
            .expect("the decode-shaped layer should evaluate");
        if step >= DECODE_WARMUP_STEPS {
            samples.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        }
        model.advance_probe_state(&mut state, 1);
    }
    let median_millis = median(&mut samples);
    eprintln!(
        "[prism-hadamard-perf:tier=layer] layer={layer_label} index={layer_index} shape=decode tokens=1 prefix_tokens={PREFILL_CHUNK_TOKENS} median_ms={median_millis:.3}",
    );
    assert!(median_millis.is_finite() && median_millis > 0.0);
}
