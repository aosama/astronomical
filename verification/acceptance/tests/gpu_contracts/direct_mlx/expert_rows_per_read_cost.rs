//! What a routed-expert projection costs by how its rows are laid out: one
//! row a gathered slot (how a draft round's rows are verified today) against
//! several rows on one gathered expert (an expert read once for every row
//! that picked it).
//!
//! Random 6-bit weights of the 35B-A3B's fused gate and up projection
//! (1,024 outputs over 2,048 inputs), 40 projections an evaluation as a
//! forward's 40 layers, so the launch overhead and the weight bytes streamed
//! are the real pass's. It prices the premise before any kernel is written:
//! if four rows on one expert cost about what one row does, grouping a
//! round's rows by expert saves what the route overlap promises.

use std::time::Instant;

use mlx_bridge::MlxArray;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const EXPERT_COUNT: i32 = 48;
const OUTPUT_WIDTH: i32 = 1_024;
const INPUT_WIDTH: i32 = 2_048;
const QUANTIZATION_BITS: i32 = 6;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const LAYERS_PER_EVALUATION: usize = 40;
const TIMED_EVALUATIONS: usize = 12;

struct QuantizedExperts {
    packed_weights: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn quantized_experts(runtime: &MlxRuntime) -> QuantizedExperts {
    let value_count = (EXPERT_COUNT * OUTPUT_WIDTH * INPUT_WIDTH) as usize;
    let weights = runtime
        .array_from_f32(
            &super::synthetic_values::pseudo_random_values(value_count, 7),
            &[EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH],
        )
        .expect("the weights upload");
    let (packed_weights, scales, biases) = runtime
        .quantize_affine(&weights, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)
        .expect("the weights quantize");
    runtime
        .evaluate_arrays(&[&packed_weights, &scales, &biases])
        .expect("the quantized weights evaluate");
    QuantizedExperts {
        packed_weights,
        scales,
        biases,
    }
}

/// Median milliseconds of an evaluation of 40 gathered projections, each of
/// `slot_count` gathered experts with `rows_per_slot` rows on every one.
fn median_milliseconds(
    runtime: &MlxRuntime,
    experts: &QuantizedExperts,
    slot_count: i32,
    rows_per_slot: i32,
) -> f64 {
    let activation_values = super::synthetic_values::pseudo_random_values(
        (slot_count * rows_per_slot * INPUT_WIDTH) as usize,
        11,
    );
    let activations = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &activation_values,
                    &[slot_count, rows_per_slot, INPUT_WIDTH],
                )
                .expect("the activations upload"),
            mlx_bridge::MlxDtype::BFloat16,
        )
        .expect("the activations convert");
    let mut layer_expert_ids = Vec::with_capacity(LAYERS_PER_EVALUATION);
    for layer_index in 0..LAYERS_PER_EVALUATION as i32 {
        let mut expert_ids = Vec::with_capacity(slot_count as usize);
        for slot_index in 0..slot_count {
            expert_ids.push((slot_index + layer_index) % EXPERT_COUNT);
        }
        expert_ids.sort_unstable();
        layer_expert_ids.push(
            runtime
                .array_from_i32(&expert_ids, &[slot_count])
                .expect("the expert ids upload"),
        );
    }
    let mut evaluation_milliseconds = Vec::with_capacity(TIMED_EVALUATIONS);
    for evaluation_index in 0..TIMED_EVALUATIONS + 2 {
        let started_at = Instant::now();
        let mut projections = Vec::with_capacity(LAYERS_PER_EVALUATION);
        for expert_ids in &layer_expert_ids {
            projections.push(
                runtime
                    .gather_quantized_matmul_affine(
                        &activations,
                        &experts.packed_weights,
                        &experts.scales,
                        &experts.biases,
                        None,
                        Some(expert_ids),
                        true,
                        QUANTIZATION_GROUP_SIZE,
                        QUANTIZATION_BITS,
                        true,
                    )
                    .expect("the gathered projection builds"),
            );
        }
        let projection_references: Vec<&MlxArray> = projections.iter().collect();
        runtime
            .evaluate_arrays(&projection_references)
            .expect("the projections evaluate");
        // The first two evaluations compile and warm the kernel.
        if evaluation_index >= 2 {
            evaluation_milliseconds.push(started_at.elapsed().as_secs_f64() * 1000.0);
        }
    }
    evaluation_milliseconds.sort_by(f64::total_cmp);
    evaluation_milliseconds[evaluation_milliseconds.len() / 2]
}

#[test]
#[ignore = "runs MLX on the GPU; about 1 GB of random expert weights"]
#[serial]
fn should_price_rows_on_one_gathered_expert_against_one_row_a_slot() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(6 * 1024 * 1024 * 1024, 1024 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let experts = quantized_experts(&runtime);
    // (slots, rows a slot, what the layout stands for)
    let layouts = [
        (8, 1, "1 row, 8 experts: a plain step"),
        (32, 1, "4 rows x 8 experts, one row a slot: today's round"),
        (20, 1, "20 slots of one row: the distinct experts alone"),
        (20, 2, "20 experts, 2 rows each"),
        (20, 4, "20 experts, 4 rows each: grouped and padded"),
        (8, 4, "8 experts, 4 rows each: every row shares all experts"),
        (48, 1, "6 rows x 8 experts, one row a slot"),
        (27, 4, "27 experts, 4 rows each: 6 rows grouped and padded"),
    ];
    for (slot_count, rows_per_slot, description) in layouts {
        let milliseconds = median_milliseconds(&runtime, &experts, slot_count, rows_per_slot);
        eprintln!(
            "[expert-rows-per-read] {slot_count:>2} slots x {rows_per_slot} rows: \
             {milliseconds:6.2} ms for 40 projections ({description})"
        );
    }
}
