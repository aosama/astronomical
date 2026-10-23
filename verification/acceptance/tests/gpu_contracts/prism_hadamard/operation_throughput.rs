//! Tier 1: the individual GPU operations of the folded ternary path.
//!
//! The Hadamard activation transform and the rotated ternary projection
//! launches are timed at the two shapes serving runs them at: one decode row
//! and one 2,048-token prefill chunk. Fused sibling projections are timed as
//! the single stacked launch the forward pass issues. Inputs are real
//! activations: the packed embedding of Romeo and Juliet tokens, and the
//! first layer's SwiGLU intermediate for the intermediate-width transform
//! and down projection.

use astronomical_model_serving::prism_hadamard_qwen35::model::{
    PrismHadamardProbeProjection, PrismHadamardTextModel,
};
use mlx_c_rust::MlxArray;
use serial_test::serial;

use super::support::{
    PREFILL_CHUNK_TOKENS, loaded_journey_model, materialized, measure_iterations, measure_streamed,
    median, pack_tokenizer, romeo_and_juliet_prompt_token_ids,
};

const WARMUP_ITERATIONS: usize = 3;
const MEASURED_ITERATIONS: usize = 10;
/// Decode-shaped operations finish in well under a millisecond, where one
/// host synchronization per call dominates; streaming this many copies per
/// evaluation reports the kernel's own cost beside the per-call figure.
const STREAMED_REPETITIONS: usize = 16;
/// The ternary group size and bit width the pack's packed matrices use.
const QUANTIZATION_GROUP_SIZE: usize = 128;
const QUANTIZATION_BITS: usize = 2;
/// Scales and biases are float16: two bytes each per group.
const SCALE_AND_BIAS_BYTES_PER_GROUP: usize = 4;

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and measures the Hadamard transform and folded ternary matmul on the graphics processor"]
fn should_measure_hadamard_transform_and_folded_ternary_matmul_throughput() {
    let model = loaded_journey_model();
    let tokenizer = pack_tokenizer();
    let dimensions = model.probe_dimensions();
    let hidden_width = dimensions.hidden_width;
    let intermediate_width = dimensions.intermediate_width;
    let gated_delta_input_projection_rows = dimensions.gated_delta_input_projection_rows;
    let dense_mlp_gate_and_up_rows = dimensions.dense_mlp_gate_and_up_rows;
    let vocabulary_size = dimensions.vocabulary_size;
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&tokenizer, PREFILL_CHUNK_TOKENS);

    for (shape_label, token_count) in [("decode", 1_usize), ("prefill", PREFILL_CHUNK_TOKENS)] {
        let hidden_states = materialized(
            &model,
            model
                .probe_embedding(&prompt_token_ids[..token_count])
                .expect("the Romeo and Juliet tokens should embed"),
        );
        let intermediate_states = materialized(
            &model,
            model
                .probe_dense_mlp_intermediate(&hidden_states)
                .expect("the first layer's MLP intermediate should build"),
        );

        report_transform(
            &model,
            shape_label,
            token_count,
            hidden_width,
            &hidden_states,
        );
        report_transform(
            &model,
            shape_label,
            token_count,
            intermediate_width,
            &intermediate_states,
        );
        report_projection(
            &model,
            shape_label,
            "gated_delta_query_key_value_and_gate",
            PrismHadamardProbeProjection::GatedDeltaQueryKeyValueAndGate,
            token_count,
            gated_delta_input_projection_rows,
            hidden_width,
            &hidden_states,
        );
        report_projection(
            &model,
            shape_label,
            "dense_mlp_gate_and_up",
            PrismHadamardProbeProjection::DenseMlpGateAndUp,
            token_count,
            dense_mlp_gate_and_up_rows,
            hidden_width,
            &hidden_states,
        );
        report_projection(
            &model,
            shape_label,
            "dense_mlp_down",
            PrismHadamardProbeProjection::DenseMlpDown,
            token_count,
            hidden_width,
            intermediate_width,
            &intermediate_states,
        );
        if token_count == 1 {
            // Serving projects only the final row to the vocabulary, so the
            // language-model head has no prefill-shaped measurement.
            report_projection(
                &model,
                shape_label,
                "language_model_head",
                PrismHadamardProbeProjection::LanguageModelHead,
                token_count,
                vocabulary_size,
                hidden_width,
                &hidden_states,
            );
        }
    }
}

fn report_transform(
    model: &PrismHadamardTextModel,
    shape_label: &str,
    token_count: usize,
    width: usize,
    activations: &MlxArray,
) {
    let mut samples = measure_iterations(model, WARMUP_ITERATIONS, MEASURED_ITERATIONS, || {
        model
            .probe_forward_transform(activations)
            .expect("the Hadamard transform should build")
    });
    let median_millis = median(&mut samples);
    let mut streamed_samples = measure_streamed(
        model,
        STREAMED_REPETITIONS,
        WARMUP_ITERATIONS,
        MEASURED_ITERATIONS,
        || {
            model
                .probe_forward_transform(activations)
                .expect("the Hadamard transform should build")
        },
    );
    let streamed_median_millis = median(&mut streamed_samples);
    // The transform reads float16 in and writes float16 out through a float32
    // working domain; count the activation payload moved once each way.
    let activation_bytes = (token_count * width * 2) as f64;
    eprintln!(
        "[prism-hadamard-perf:tier=operation] op=hadamard_transform shape={shape_label} tokens={token_count} width={width} median_ms={median_millis:.3} streamed_median_ms={streamed_median_millis:.3} activation_mb_per_ms={:.1}",
        2.0 * activation_bytes / (1024.0 * 1024.0) / streamed_median_millis,
    );
    assert!(median_millis.is_finite() && median_millis > 0.0);
}

#[allow(clippy::too_many_arguments)]
fn report_projection(
    model: &PrismHadamardTextModel,
    shape_label: &str,
    projection_label: &str,
    projection: PrismHadamardProbeProjection,
    token_count: usize,
    output_rows: usize,
    input_width: usize,
    activations: &MlxArray,
) {
    let mut samples = measure_iterations(model, WARMUP_ITERATIONS, MEASURED_ITERATIONS, || {
        model
            .probe_packed_projection(projection, activations)
            .expect("the rotated projection should build")
    });
    let median_millis = median(&mut samples);
    let mut streamed_samples = measure_streamed(
        model,
        STREAMED_REPETITIONS,
        WARMUP_ITERATIONS,
        MEASURED_ITERATIONS,
        || {
            model
                .probe_packed_projection(projection, activations)
                .expect("the rotated projection should build")
        },
    );
    let streamed_median_millis = median(&mut streamed_samples);
    let packed_weight_bytes = output_rows * input_width * QUANTIZATION_BITS / 8
        + output_rows * (input_width / QUANTIZATION_GROUP_SIZE) * SCALE_AND_BIAS_BYTES_PER_GROUP;
    let weight_gigabytes_per_second = packed_weight_bytes as f64
        / (1024.0 * 1024.0 * 1024.0)
        / (streamed_median_millis / 1_000.0);
    let giga_operations_per_second =
        2.0 * token_count as f64 * output_rows as f64 * input_width as f64
            / 1.0e9
            / (streamed_median_millis / 1_000.0);
    eprintln!(
        "[prism-hadamard-perf:tier=operation] op=packed_projection projection={projection_label} shape={shape_label} tokens={token_count} rows={output_rows} width={input_width} median_ms={median_millis:.3} streamed_median_ms={streamed_median_millis:.3} weight_gb_per_second={weight_gigabytes_per_second:.1} gflop_per_second={giga_operations_per_second:.1}",
    );
    assert!(median_millis.is_finite() && median_millis > 0.0);
}
