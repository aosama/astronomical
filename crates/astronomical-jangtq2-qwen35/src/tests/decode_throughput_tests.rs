//! Bottom-tier throughput measurements for the jangtq2 graphics-processor
//! codebook decoder.
//!
//! This is the lowest layer of the jangtq2 serving call stack (the decode
//! kernels in `mlx_decode`); per the repository's performance instructions,
//! throughput measurement starts here and climbs upward, so a regression or
//! improvement is attributable to one layer before the layers above it are
//! measured. The tests report `median_ms`, `decoded_gigabytes_per_second`,
//! and the per-shape peak-reservation arithmetic (packed source plus
//! decoded output); they assert measurement shape (finite,
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
    // The complete-layer cases allocate the packed source plus a 1.5 GB
    // decoded projection; the limit stays well above that so the lane also
    // measures a regressed decoder that reintroduced workspace arrays.
    MlxRuntime::initialize(
        MlxMemoryLimits::new(20 * 1024 * 1024 * 1024, 20 * 1024 * 1024 * 1024)
            .expect("the jangtq2 decode throughput memory limits should be valid"),
    )
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
/// The decoder evaluates and detaches its output, so each call is already
/// fully synchronous and per-call wall time is honest without extra evaluation.
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
        .array_from_f32(&row_scales_host, &[expert_count as i32, output_rows as i32])
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
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::decode_expert_weight_rows(
                &runtime,
                &packed_code_words,
                &row_scales,
                bit_width,
            )
            .expect("the warmup decode should succeed"),
        );
    }
    let mut measured_durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut measured_peak_delta_bytes: Option<u64> = None;
    for measured_iteration in 0..MEASURED_ITERATIONS {
        if measured_iteration == 0 {
            // Peak-memory contract: the decoder must live inside the
            // admission reserve (packed source plus persisted output). If a
            // change reintroduces intermediate arrays, page admission would
            // under-reserve real serving: fail here instead.
            runtime
                .reset_peak_memory()
                .expect("the MLX peak-memory counter should reset");
        }
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
        if measured_iteration == 0 {
            measured_peak_delta_bytes = runtime
                .memory_snapshot()
                .ok()
                .map(|memory_snapshot| memory_snapshot.peak_memory_bytes() as u64);
        }
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
    if let Some(observed_peak_delta_bytes) = measured_peak_delta_bytes {
        // The bound mirrors expert_page_peak_allocation_bytes for this page:
        // packed sources plus the persisted output, with 20 percent slack for
        // MLX-internal buffers the decoder cannot see. The Metal decoder has
        // no workspace term, so any temporaries an MLX upgrade introduced
        // would fail here before admission under-reserved real serving.
        let packed_source_bytes =
            (expert_count * output_rows * packed_words * std::mem::size_of::<u32>()) as u64;
        let persisted_output_bytes = decoded_element_count * std::mem::size_of::<u16>() as u64;
        let bound_bytes = packed_source_bytes + persisted_output_bytes;
        let bound_bytes_with_slack = bound_bytes + bound_bytes / 5;
        assert!(
            observed_peak_delta_bytes <= bound_bytes_with_slack,
            "case={case_name} decode peaked at {observed_peak_delta_bytes} bytes, above the admission-bound-plus-slack {bound_bytes_with_slack}: the page decoder allocated intermediate arrays the admission contract does not price"
        );
        eprintln!(
            "[jangtq2-decode-perf:tier=operation] case={case_name} peak_delta_bytes={observed_peak_delta_bytes} admission_bound_bytes={bound_bytes}"
        );
    }
    eprintln!(
        "[jangtq2-decode-perf:tier=operation] case={case_name} experts={expert_count} rows={output_rows} packed_words={packed_words} bits={bit_width} median_ms={median_ms:.3} decoded_gb={decoded_gigabytes:.3} decoded_gb_per_second={decoded_gigabytes_per_second:.3}"
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
            let rotated_activations =
                mlx_decode::rotate_activations_hadamard32(&runtime, &activations)
                    .expect("the measured rotation should succeed");
            measured_durations_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
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

/// One decode token's routed projection, both ways: the materializing path
/// (`decode_expert_weight_rows` + lazy transpose + `gather_mm`) and the fused
/// gathered projection kernel, each timed to a fully evaluated output so the
/// two medians are directly comparable. Reports the ratio so the decode-token
/// cost model (three projections per layer, 48 layers) can be read straight
/// off the tier-0 line.
#[cfg(feature = "direct-mlx")]
fn measure_fused_gathered_projection_throughput(
    case_name: &str,
    packed_shape: &[i32],
    bit_width: u32,
) {
    use super::super::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel;

    let runtime = test_runtime();
    let fused_kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let expert_count = packed_shape[0] as usize;
    let output_rows = packed_shape[1] as usize;
    let packed_words = packed_shape[2] as usize;
    let input_width = packed_words * 32 / bit_width as usize;
    let packed_code_words = MlxArray::from_u32(
        &deterministic_packed_words(
            u64::try_from(expert_count * 31 + packed_words).unwrap_or(0xA5A5),
            expert_count * output_rows * packed_words,
        ),
        packed_shape,
    )
    .expect("the synthetic packed page should form a valid uint32 tensor");
    let row_scales = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &deterministic_row_scales(
                        u64::try_from(expert_count * 17 + output_rows).unwrap_or(0x5A5A),
                        expert_count * output_rows,
                    ),
                    &[expert_count as i32, output_rows as i32],
                )
                .expect("the synthetic row scales should upload"),
            MlxDtype::Float16,
        )
        .expect("the synthetic row scales should convert to float16");
    let activations_host: Vec<f32> = (0..input_width)
        .map(|element_index| ((element_index % 33) as f32 - 16.0) / 16.0)
        .collect();
    let hidden = runtime
        .astype(
            &runtime
                .array_from_f32(&activations_host, &[1, input_width as i32])
                .expect("the activation fixture should upload"),
            MlxDtype::BFloat16,
        )
        .expect("the activations should convert to bfloat16");
    let route_host: Vec<i32> = (0..expert_count as i32).collect();
    let flat_route = MlxArray::from_i32(&route_host, &[expert_count as i32])
        .expect("the flat route should upload");
    let gather_route = MlxArray::from_i32(&route_host, &[1, expert_count as i32])
        .expect("the gather route should upload");
    let expanded_hidden = runtime
        .reshape(&hidden, &[1, 1, 1, input_width as i32])
        .expect("the activations should expand for gather_mm");
    runtime
        .evaluate_arrays(&[
            &packed_code_words,
            &row_scales,
            &hidden,
            &flat_route,
            &gather_route,
            &expanded_hidden,
        ])
        .expect("the inputs should materialize before timing");

    let materializing_path = || {
        let decoded_page = mlx_decode::decode_expert_weight_rows(
            &runtime,
            &packed_code_words,
            &row_scales,
            bit_width,
        )
        .expect("the materializing decode should succeed");
        let transposed_page = runtime
            .transpose_axes(&decoded_page, &[0, 2, 1])
            .expect("the decoded page should transpose");
        let gathered = runtime
            .gather_dense_matmul(
                &expanded_hidden,
                &transposed_page,
                None,
                Some(&gather_route),
                false,
            )
            .expect("gather_mm should build");
        runtime
            .evaluate_arrays(&[&gathered])
            .expect("the gathered projection should evaluate");
    };
    let fused_path = || {
        let projected = fused_kernel
            .apply(
                &runtime,
                &hidden,
                &flat_route,
                &packed_code_words,
                &row_scales,
                bit_width,
            )
            .expect("the fused projection should launch");
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the fused projection should evaluate");
    };
    let measure = |path: &dyn Fn()| -> f64 {
        for _ in 0..WARMUP_ITERATIONS {
            path();
        }
        let mut durations_ms = Vec::with_capacity(MEASURED_ITERATIONS);
        for _ in 0..MEASURED_ITERATIONS {
            let started_at = std::time::Instant::now();
            path();
            durations_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        }
        median_duration_millis(&durations_ms)
    };
    let materializing_median_ms = measure(&materializing_path);
    let fused_median_ms = measure(&fused_path);
    assert!(
        materializing_median_ms.is_finite() && materializing_median_ms > 0.0,
        "the materializing median must be a positive measurement"
    );
    assert!(
        fused_median_ms.is_finite() && fused_median_ms > 0.0,
        "the fused median must be a positive measurement"
    );
    eprintln!(
        "[jangtq2-decode-perf:tier=operation] case={case_name} experts={expert_count} rows={output_rows} packed_words={packed_words} bits={bit_width} materializing_decode_plus_gather_mm_median_ms={materializing_median_ms:.3} fused_gathered_projection_median_ms={fused_median_ms:.3} fused_speedup={:.2}",
        materializing_median_ms / fused_median_ms
    );
}

#[cfg(feature = "direct-mlx")]
#[test]
#[serial]
#[ignore = "graphics-processor throughput measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_decode_step_top_k_page_fused_gathered_projection_throughput() {
    measure_fused_gathered_projection_throughput(
        "top8_gate_up_2bit_fused",
        &[
            TOP_K_EXPERT_COUNT as i32,
            GATE_UP_OUTPUT_ROWS as i32,
            GATE_UP_PACKED_WORDS_2BIT as i32,
        ],
        TWO_BIT_WIDTH,
    );
    measure_fused_gathered_projection_throughput(
        "top8_down_2bit_fused",
        &[
            TOP_K_EXPERT_COUNT as i32,
            DOWN_OUTPUT_ROWS as i32,
            DOWN_PACKED_WORDS_2BIT as i32,
        ],
        TWO_BIT_WIDTH,
    );
    measure_fused_gathered_projection_throughput(
        "top8_down_3bit_fused",
        &[
            TOP_K_EXPERT_COUNT as i32,
            DOWN_OUTPUT_ROWS as i32,
            DOWN_PACKED_WORDS_3BIT as i32,
        ],
        THREE_BIT_WIDTH,
    );
}
