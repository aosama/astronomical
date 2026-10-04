use std::time::Instant;

use astronomical_runtime_integration::MlxRuntime;

use crate::common::runtime_test_support::runtime;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const WARMUP_ITERATIONS: usize = 4;
const MEASUREMENT_ITERATIONS: usize = 5;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// Ornith-1.5-35B-A3B attention projection shapes: (output_rows, input_columns).
const PROJECTION_SHAPES: [(usize, usize); 3] = [(8_192, 2_048), (4_096, 2_048), (2_048, 4_096)];
const TOKEN_COUNTS: [usize; 4] = [512, 1_024, 2_048, 4_096];

/// Tile candidates for the Qwen3.5 prefill NAX path plus the stock MLX tile,
/// swept through the ASTRONOMICAL_QMM_NAX_TILE override.
/// Tuple order is (bm, bn, bk, wm, wn). The instantiation macro takes
/// (bm, bk, bn), so a (64, 32, 64) label is bk=32, not bn=32; bn=32
/// (TN=1 NAX tile) produces wrong output on the stock kernel and is not swept.
const TILE_CANDIDATES: [(usize, usize, usize, usize, usize); 5] = [
    (64, 64, 64, 2, 2),
    (128, 64, 64, 2, 2),
    (64, 128, 64, 2, 2),
    (64, 64, 32, 2, 2),
    (64, 64, 64, 4, 1),
];

#[test]
#[ignore = "measures NAX qmm tile variants on real GPU kernels; run via scripts/run-bounded-cargo-test.sh"]
fn should_measure_nax_qmm_tile_variants_for_transposed_prefill_shapes() {
    let runtime = runtime();

    for (output_rows, input_columns) in PROJECTION_SHAPES {
        let quantized_weights = build_quantized_weights(&runtime, output_rows, input_columns);
        for token_count in TOKEN_COUNTS {
            let activations = build_activations(&runtime, token_count, input_columns);
            let stock_output = build_with_tile(
                &runtime,
                &activations,
                &quantized_weights,
                TILE_CANDIDATES[0],
            );
            let stock_values = materialize_output(&runtime, &stock_output);
            let stock_magnitude = stock_values
                .iter()
                .fold(0.0f32, |largest, value| largest.max(value.abs()));

            for tile_candidate in TILE_CANDIDATES {
                let median_millis =
                    measure_tile(&runtime, &activations, &quantized_weights, tile_candidate);
                let variant_output =
                    build_with_tile(&runtime, &activations, &quantized_weights, tile_candidate);
                let variant_values = materialize_output(&runtime, &variant_output);
                let maximum_absolute_error = stock_values
                    .iter()
                    .zip(variant_values.iter())
                    .fold(0.0f32, |largest, (stock, variant)| {
                        largest.max((stock - variant).abs())
                    });
                let relative_error =
                    maximum_absolute_error / stock_magnitude.max(f32::MIN_POSITIVE);
                assert!(
                    relative_error < 1e-2,
                    "tile {tile_candidate:?} diverged from the stock tile on shape ({output_rows},{input_columns}) tokens={token_count}: relative_error={relative_error}"
                );
                eprintln!(
                    "[qmm-tile-sweep:result] shape=({output_rows},{input_columns}) tokens={token_count} tile=({},{},{},{},{}) median_ms={median_millis:.3} relative_error={relative_error:.2e}",
                    tile_candidate.0,
                    tile_candidate.1,
                    tile_candidate.2,
                    tile_candidate.3,
                    tile_candidate.4,
                );
            }
        }
    }

    // SAFETY: scripts/run-bounded-cargo-test.sh pins --test-threads=1 for this
    // binary, so no other test thread can observe the environment mid-sweep.
    unsafe { std::env::remove_var("ASTRONOMICAL_QMM_NAX_TILE") };
}

struct QuantizedWeights {
    packed_words: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn build_quantized_weights(
    runtime: &MlxRuntime,
    output_rows: usize,
    input_columns: usize,
) -> QuantizedWeights {
    let words_per_row = input_columns * BITS as usize / 32;
    let weight_words = deterministic_weight_words(output_rows * words_per_row);
    let packed_words = runtime
        .array_from_u32(&weight_words, &[output_rows as i32, words_per_row as i32])
        .expect("the packed quantized weights should be valid");

    let groups_per_row = input_columns / GROUP_SIZE as usize;
    let scale_values = vec![0.01f32; output_rows * groups_per_row];
    let float32_scales = runtime
        .array_from_f32(&scale_values, &[output_rows as i32, groups_per_row as i32])
        .expect("the affine scales should be valid");
    let scales = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the scales should cast to float16");
    let biases = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the zero biases should cast to float16");

    runtime
        .evaluate_arrays(&[&packed_words, &scales, &biases])
        .expect("the quantized weights should materialize");
    QuantizedWeights {
        packed_words,
        scales,
        biases,
    }
}

fn build_activations(runtime: &MlxRuntime, token_count: usize, input_columns: usize) -> MlxArray {
    let activation_values = deterministic_activation_values(token_count * input_columns);
    let float32_activations = runtime
        .array_from_f32(
            &activation_values,
            &[token_count as i32, input_columns as i32],
        )
        .expect("the activation matrix should be valid");
    let activations = runtime
        .astype(&float32_activations, MlxDtype::Float16)
        .expect("the activations should cast to float16");
    runtime
        .evaluate_arrays(&[&activations])
        .expect("the activations should materialize");
    activations
}

fn build_with_tile(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    quantized_weights: &QuantizedWeights,
    tile_candidate: (usize, usize, usize, usize, usize),
) -> MlxArray {
    apply_tile_override(tile_candidate);
    runtime
        .quantized_matmul_affine(
            activations,
            &quantized_weights.packed_words,
            &quantized_weights.scales,
            &quantized_weights.biases,
            true,
            GROUP_SIZE,
            BITS,
        )
        .expect("the transposed affine quantized matmul should build")
}

fn measure_tile(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    quantized_weights: &QuantizedWeights,
    tile_candidate: (usize, usize, usize, usize, usize),
) -> f64 {
    let (bm, bn, bk, wm, wn) = tile_candidate;
    for warmup_iteration in 1..=WARMUP_ITERATIONS {
        eprintln!(
            "[qmm-tile-sweep] warmup {warmup_iteration}/{WARMUP_ITERATIONS} tile=({bm},{bn},{bk},{wm},{wn})",
        );
        let product = build_with_tile(runtime, activations, quantized_weights, tile_candidate);
        evaluate_product(runtime, &product);
    }
    let mut elapsed_millis = Vec::with_capacity(MEASUREMENT_ITERATIONS);
    for measurement_iteration in 1..=MEASUREMENT_ITERATIONS {
        eprintln!(
            "[qmm-tile-sweep] measurement {measurement_iteration}/{MEASUREMENT_ITERATIONS} tile=({bm},{bn},{bk},{wm},{wn})",
        );
        let product = build_with_tile(runtime, activations, quantized_weights, tile_candidate);
        let started_at = Instant::now();
        evaluate_product(runtime, &product);
        elapsed_millis.push(started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    runtime
        .clear_allocator_cache()
        .expect("the sweep should release reclaimable MLX allocations");
    elapsed_millis.sort_by(f64::total_cmp);
    let middle_index = elapsed_millis.len() / 2;
    (elapsed_millis[middle_index - 1] + elapsed_millis[middle_index]) / 2.0
}

fn apply_tile_override(tile_candidate: (usize, usize, usize, usize, usize)) {
    let (bm, bn, bk, wm, wn) = tile_candidate;
    // SAFETY: scripts/run-bounded-cargo-test.sh pins --test-threads=1 for this
    // binary, so no other test thread can observe the environment mid-sweep.
    unsafe {
        std::env::set_var(
            "ASTRONOMICAL_QMM_NAX_TILE",
            format!("{bm},{bn},{bk},{wm},{wn}"),
        );
    }
}

fn evaluate_product(runtime: &MlxRuntime, product: &MlxArray) {
    runtime
        .evaluate_arrays(&[product])
        .expect("the quantized product should evaluate");
}

fn materialize_output(runtime: &MlxRuntime, output: &MlxArray) -> Vec<f32> {
    let float32_output = runtime
        .astype(output, MlxDtype::Float32)
        .expect("the output should cast to float32");
    float32_output
        .to_vec_f32()
        .expect("the output should read back as float32")
}

fn deterministic_activation_values(element_count: usize) -> Vec<f32> {
    let mut generator_state = 0x2545_F491_4F6C_DD1Du64;
    (0..element_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit_interval = ((generator_state >> 33) as f32) / ((1u32 << 31) as f32);
            unit_interval * 2.0 - 1.0
        })
        .collect()
}

fn deterministic_weight_words(word_count: usize) -> Vec<u32> {
    let mut generator_state = 0x9E37_79B9_7F4A_7C15u64;
    (0..word_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (generator_state >> 32) as u32
        })
        .collect()
}
