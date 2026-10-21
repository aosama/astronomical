//! Bottom-tier throughput measurements for the jangtq2 graphics-processor
//! codebook decoder.
//!
//! This is the lowest layer of the jangtq2 serving call stack (the decode
//! kernels in `mlx_decode`); per the repository's performance instructions,
//! throughput measurement starts here and climbs upward, so a regression or
//! improvement is attributable to one layer before the layers above it are
//! measured. The tests report `median_ms`, `decoded_gigabytes_per_second`,
//! and the per-shape peak-reservation arithmetic implied by the decoder's
//! exported workspace constants; they assert measurement shape (finite,
//! positive, exact element counts) and never assert throughput thresholds —
//! the durable throughput ledger owns trend comparisons.
//!
//! Run through an optimized build with `cargo test-jangtq2-decode-throughput`
//! (see `.cargo/config.toml`): the decoder is memory-traffic bound, so
//! debug-build numbers understate every ratio and must not be recorded. Every
//! test is `#[serial]` because MLX graphics-processor tests never run in
//! parallel.

use super::super::codebook;
#[cfg(feature = "direct-mlx")]
use super::super::mlx_decode;
#[cfg(feature = "direct-mlx")]
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::MlxMemoryLimits;
#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxRuntime;
#[cfg(feature = "direct-mlx")]
use serial_test::serial;

/// Routed experts serving one decode token (Qwen3.5 `num_experts_per_tok`).
const TOP_K_EXPERT_COUNT: usize = 8;
/// Every routed expert of one layer, as prefill complete-layer streaming loads.
const COMPLETE_LAYER_EXPERT_COUNT: usize = 256;
/// Mixture-of-Experts intermediate size: gate and up output rows.
const GATE_UP_OUTPUT_ROWS: usize = 1024;
/// Hidden size: down projection output rows (and every projection input width).
const DOWN_OUTPUT_ROWS: usize = 3072;
/// Packed uint32 words per gate/up row at 2 bits (3072 inputs × 2 bits / 32).
const GATE_UP_PACKED_WORDS_2BIT: usize = 192;
/// Packed uint32 words per down row at 2 bits (1024 inputs × 2 bits / 32).
const DOWN_PACKED_WORDS_2BIT: usize = 64;
/// Packed uint32 words per down row at 3 bits (1024 inputs × 3 bits / 32).
const DOWN_PACKED_WORDS_3BIT: usize = 96;
const TWO_BIT_WIDTH: u32 = 2;
const THREE_BIT_WIDTH: u32 = 3;
const GIGABYTE_BYTES: f64 = 1_073_741_824.0;
/// Discarded work before timing so first-touch page faults and Metal pipeline
/// compilation costs never contaminate the measured medians.
const WARMUP_ITERATIONS: usize = 3;
const MEASURED_ITERATIONS: usize = 10;
/// Prefill prompt-processing chunk token count the repository pins for every
/// throughput measurement, used here only to size the rotation input.
const PREFILL_CHUNK_TOKENS: usize = 2_048;

fn test_runtime() -> astronomical_runtime_integration::MlxRuntime {
    // The complete-layer cases must admit the decoder's current whole-page
    // workspace (roughly 15 GB before expert blocking, a few GB after), so the
    // limit stays above both the baseline and the remediated peak: the same
    // test then measures the code before and after the fix without edits.
    MlxRuntime::initialize(MlxMemoryLimits::new(
        20 * 1024 * 1024 * 1024,
        20 * 1024 * 1024 * 1024,
    )
    .expect("the jangtq2 decode throughput memory limits should be valid"))
    .expect("the direct MLX runtime should initialize")
}

/// Deterministic splitmix64 packed words: throughput fixtures must be
/// reproducible run over run so medians compare across code changes. These are
/// weight bit patterns, not language input, so a seeded generator (rather than
/// the Romeo and Juliet fixture, which serves prompt-level journeys) is the
/// reproducible source here.
fn deterministic_packed_words(seed: u64, word_count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..word_count)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (mixed ^ (mixed >> 31)) as u32
        })
        .collect()
}

/// Row scales exactly representable in float16 so the fixture never depends on
/// rounding of a wider dtype.
fn deterministic_row_scales(seed: u64, row_count: usize) -> Vec<f32> {
    (0..row_count)
        .map(|row_index| {
            let step = seed.wrapping_add(row_index as u64) % 16;
            ((step + 4) as f32) / 8.0
        })
        .collect()
}

fn median_duration_millis(durations_ms: &[f64]) -> f64 {
    let mut sorted_durations = durations_ms.to_vec();
    sorted_durations.sort_by(|left, right| left.total_cmp(right));
    sorted_durations[sorted_durations.len() / 2]
}

/// Runs one decode case: warms up, measures, and reports the tier-0 line.
///
/// The decoder detaches every staged array, so each call is already fully
/// synchronous and per-call wall time is honest without extra evaluation.
#[cfg(feature = "direct-mlx")]
fn measure_decode_throughput(case_name: &str, packed_shape: &[i32], bit_width: u32) {
    let runtime = test_runtime();
    let expert_count = packed_shape[0] as usize;
    let output_rows = packed_shape[1] as usize;
    let packed_words = packed_shape[2] as usize;
    let packed_words_host = deterministic_packed_words(
        u64::try_from(expert_count * 31 + packed_words).unwrap_or(0xA5A5),
        expert_count * output_rows * packed_words,
    );
    let row_scales_host = deterministic_row_scales(
        u64::try_from(expert_count * 17 + output_rows).unwrap_or(0x5A5A),
        expert_count * output_rows,
    );
    let packed_code_words = MlxArray::from_u32(&packed_words_host, packed_shape)
        .expect("the synthetic packed page should form a valid uint32 tensor");
    let row_scales_float32 = runtime
        .array_from_f32(
            &row_scales_host,
            &[expert_count as i32, output_rows as i32],
        )
        .expect("the synthetic row scales should upload");
    let row_scales = runtime
        .astype(&row_scales_float32, MlxDtype::Float16)
        .expect("the synthetic row scales should convert to float16");
    runtime
        .evaluate_arrays(&[&packed_code_words, &row_scales])
        .expect("the decode inputs should materialize before timing");

    let decoded_shape = codebook::decoded_weight_page_shape(
        &[expert_count, output_rows, packed_words],
        bit_width,
        expert_count,
    )
    .expect("the synthetic page shape should decode");
    let decoded_element_count: u64 = decoded_shape.iter().product::<usize>() as u64;
    let decoded_gigabytes =
        decoded_element_count as f64 * std::mem::size_of::<u16>() as f64 / GIGABYTE_BYTES;
    // Peak-reservation arithmetic the pager admits per projection: the
    // exported temporary count times four-byte elements, plus the auxiliary
    // index arrays. Reporting it beside the timing keeps the memory contract
    // visible at the tier where it is created.
    let auxiliary_bound_element_count =
        (decoded_shape[2] as u64).max((decoded_shape[0] * decoded_shape[1]) as u64);
    let block_element_count = decoded_element_count
        / u64::try_from(decoded_shape[0]).expect("the expert count fits u64")
        * u64::try_from(codebook::decode_block_expert_count(expert_count))
            .expect("the block expert count fits u64");
    let temporary_workspace_gigabytes = block_element_count as f64
        * mlx_decode::MAXIMUM_SIMULTANEOUS_DECODE_TEMPORARY_ARRAY_COUNT as f64
        * std::mem::size_of::<u32>() as f64
        / GIGABYTE_BYTES;
    let auxiliary_workspace_gigabytes = auxiliary_bound_element_count as f64
        * mlx_decode::JANGTQ2_DECODE_AUXILIARY_ARRAY_COUNT as f64
        * std::mem::size_of::<u32>() as f64
        / GIGABYTE_BYTES;

    for _ in 0..WARMUP_ITERATIONS {
        drop(mlx_decode::decode_expert_weight_rows(
            &runtime,
            &packed_code_words,
            &row_scales,
            bit_width,
        )
        .expect("the warmup decode should succeed"));
    }
    let mut measured_durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for measured_iteration in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let decoded_weight = mlx_decode::decode_expert_weight_rows(
            &runtime,
            &packed_code_words,
            &row_scales,
            bit_width,
        )
        .expect("the measured decode should succeed");
        let elapsed_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(
            decoded_weight.shape(),
            [
                expert_count as i32,
                output_rows as i32,
                decoded_shape[2] as i32
            ],
            "the decoded page shape must match the format contract"
        );
        assert_eq!(decoded_weight.dtype(), MlxDtype::BFloat16);
        measured_durations_ms.push(elapsed_ms);
        drop(decoded_weight);
        if measured_iteration == 0 {
            eprintln!(
                "[jangtq2-decode-perf:tier=operation] case={case_name} first_iteration_ms={elapsed_ms:.3}"
            );
        }
    }
    let median_ms = median_duration_millis(&measured_durations_ms);
    let decoded_gigabytes_per_second = decoded_gigabytes / (median_ms / 1_000.0);
    assert!(
        median_ms.is_finite() && median_ms > 0.0,
        "the median decode duration must be a positive measurement"
    );
    assert!(
        decoded_gigabytes_per_second.is_finite() && decoded_gigabytes_per_second > 0.0,
        "the decoded throughput must be a positive measurement"
    );
    eprintln!(
        "[jangtq2-decode-perf:tier=operation] case={case_name} experts={expert_count} rows={output_rows} packed_words={packed_words} bits={bit_width} median_ms={median_ms:.3} decoded_gb={decoded_gigabytes:.3} decoded_gb_per_second={decoded_gigabytes_per_second:.3} temporary_workspace_gb={temporary_workspace_gigabytes:.3} auxiliary_workspace_gb={auxiliary_workspace_gigabytes:.3}"
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
#[serial]
#[ignore = "graphics-processor throughput measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_decode_step_top_k_page_codebook_decode_throughput() {
    measure_decode_throughput(
        "top8_gate_up_2bit",
        &[
            TOP_K_EXPERT_COUNT as i32,
            GATE_UP_OUTPUT_ROWS as i32,
            GATE_UP_PACKED_WORDS_2BIT as i32,
        ],
        TWO_BIT_WIDTH,
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
#[serial]
#[ignore = "graphics-processor throughput measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_complete_layer_gate_projection_codebook_decode_throughput() {
    measure_decode_throughput(
        "complete256_gate_up_2bit",
        &[
            COMPLETE_LAYER_EXPERT_COUNT as i32,
            GATE_UP_OUTPUT_ROWS as i32,
            GATE_UP_PACKED_WORDS_2BIT as i32,
        ],
        TWO_BIT_WIDTH,
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
#[serial]
#[ignore = "graphics-processor throughput measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_complete_layer_down_projection_codebook_decode_throughput() {
    measure_decode_throughput(
        "complete256_down_2bit",
        &[
            COMPLETE_LAYER_EXPERT_COUNT as i32,
            DOWN_OUTPUT_ROWS as i32,
            DOWN_PACKED_WORDS_2BIT as i32,
        ],
        TWO_BIT_WIDTH,
    );
    measure_decode_throughput(
        "complete256_down_3bit",
        &[
            COMPLETE_LAYER_EXPERT_COUNT as i32,
            DOWN_OUTPUT_ROWS as i32,
            DOWN_PACKED_WORDS_3BIT as i32,
        ],
        THREE_BIT_WIDTH,
    );
}

/// The blockwise Hadamard-32 activation rotation, at the decode-step shape
/// (one token) and the pinned prefill chunk shape, so its share of the
/// per-token cost is visible beside the decoder's.
#[cfg(feature = "direct-mlx")]
#[test]
#[serial]
#[ignore = "graphics-processor throughput measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_hadamard32_activation_rotation_throughput() {
    let runtime = test_runtime();
    for (case_name, token_count) in [
        ("rotation_decode_1_token", 1_usize),
        ("rotation_prefill_2048_tokens", PREFILL_CHUNK_TOKENS),
    ] {
        let activation_values = deterministic_row_scales(0xC0FFEE, token_count * DOWN_OUTPUT_ROWS);
        let activations = runtime
            .array_from_f32(
                &activation_values,
                &[token_count as i32, DOWN_OUTPUT_ROWS as i32],
            )
            .expect("the synthetic activations should upload");
        runtime
            .evaluate_arrays(&[&activations])
            .expect("the rotation input should materialize before timing");
        for _ in 0..WARMUP_ITERATIONS {
            drop(
                mlx_decode::rotate_activations_hadamard32(&runtime, &activations)
                    .expect("the warmup rotation should succeed"),
            );
        }
        let mut measured_durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
        for _ in 0..MEASURED_ITERATIONS {
            let started_at = std::time::Instant::now();
            let rotated_activations = mlx_decode::rotate_activations_hadamard32(
                &runtime,
                &activations,
            )
            .expect("the measured rotation should succeed");
            measured_durations_ms
                .push(started_at.elapsed().as_secs_f64() * 1_000.0);
            assert_eq!(
                rotated_activations.shape(),
                [token_count as i32, DOWN_OUTPUT_ROWS as i32]
            );
            drop(rotated_activations);
        }
        let median_ms = median_duration_millis(&measured_durations_ms);
        assert!(median_ms.is_finite() && median_ms > 0.0);
        eprintln!(
            "[jangtq2-decode-perf:tier=operation] case={case_name} tokens={token_count} width={DOWN_OUTPUT_ROWS} median_ms={median_ms:.3}"
        );
    }
}
