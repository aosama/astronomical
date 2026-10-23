//! Tier-0 prototype: prefill gathered projection straight from packed
//! jangtq2 codes, compared against the serving materializing path.
//!
//! The serving prefill path rotates activations through a four-operation MLX
//! graph, decodes the complete-layer page to a transient bfloat16 tensor
//! (about 1.6 GB per projection on the 122B bundle; the diagnostic counter
//! measured about 1 TB decoded per 10,544-token prompt), then reads that
//! tensor back once through `gather_mm`. This prototype kernel instead keeps
//! one threadgroup per (expert, 64-output-row block): it streams the
//! expert's assignments, rotates each 32-wide activation block cooperatively
//! in threadgroup memory (the blockwise Hadamard-32 is self-inverse and
//! orthonormal, and the 32-code packed group is exactly one K-tile), and
//! accumulates the dot product from registers, so neither the rotated
//! activation tensor nor the decoded weight tensor ever exists in memory.
//!
//! Prototype scope (deliberate): the kernel lives in the test tree only, the
//! activations are float32, and one thread per output row is chosen for the
//! same occupancy reason as the decode fused kernel — a production version
//! would move to simdgroup tiles once the traffic win is proven here. T-MAC's
//! published result that lookup tables lose to dequantize-in-registers on
//! graphics processors is why the codebook levels stay arithmetic.
//!
//! Measured honestly (Apple M5 Pro, release build, real complete-layer
//! shapes at the pinned 2,048-token chunk): the prototype computes exactly —
//! a mean relative error of 1e-6 against float64 host arithmetic, rotation
//! and dequant included — but runs at 0.24x to 0.50x of the materializing
//! baseline (about 97 to 177 ms versus about 43 to 49 ms per projection,
//! each stage timed separately): one scalar float32 thread per output row
//! with two threadgroup barriers per 32-code group cannot out-arithmetic
//! MLX's bfloat16 simd-group `gather_mm`, even though it eliminates the
//! 1.6 GB decoded round trip. The traffic thesis
//! survives in production evidence — the serving attribution shows the page
//! decode at 53 percent of prefill — but capturing it needs a simd-group
//! tiled kernel (bfloat16 activations, matrix-tile accumulate), not this
//! scalar layout. Two hard-won launch facts live here so they are not
//! relearned: the custom-kernel grid dimensions count THREADS, not
//! threadgroups (an undersized grid silently executes a fraction of the
//! work and reports impossibly fast timings), and every timed output must
//! stay alive across the measurement — a dropped, freed output can return a
//! stale or zeroed graph result on a later identical launch.
//!
//! Every test is `#[serial]` because graphics-processor tests never run in
//! parallel, and they are `#[ignore]`d: run them optimized through
//! `cargo test-jangtq2-decode-throughput` (the filter matches this module's
//! `..._throughput` names).

use mlx_c_rust::{MlxArray, MlxDtype, MlxMetalKernel};
use serial_test::serial;

use super::super::codebook;
use super::super::mlx_decode;
use super::super::mlx_decode::decode_expert_weight_rows;
use fixtures::{
    deterministic_packed_words, deterministic_row_scales, deterministic_token_activations,
    expert_sorted_assignments, median_duration_millis, prototype_runtime,
};
use kernel::{PREFILL_GATHERED_DEQUANT_KERNEL_SOURCE, launch_prefill_gathered_dequant};

mod fixtures;
mod kernel;

/// Routed experts serving one decode token (Qwen3.5 `num_experts_per_tok`).
const TOP_K_EXPERT_COUNT: usize = 8;
/// Every routed expert of one layer, as prefill complete-layer streaming loads.
const COMPLETE_LAYER_EXPERT_COUNT: usize = 256;
/// The prompt-processing chunk token count the repository pins for throughput
/// measurement; the assignment count is the chunk times the routed experts.
const PREFILL_CHUNK_TOKENS: usize = 2_048;
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
const WARMUP_ITERATIONS: usize = 3;
const MEASURED_ITERATIONS: usize = 10;
/// Runs one prefill-shaped comparison: the serving materializing baseline
/// (rotation graph, complete-layer page decode, `gather_mm`) against the
/// prototype kernel with the rotation folded in, each phase timed to a fully
/// evaluated output, then a parity check of the two results.
fn measure_prefill_gathered_dequant(
    case_name: &str,
    output_rows: usize,
    packed_words_per_row: usize,
    bit_width: u32,
) {
    let runtime = prototype_runtime();
    let kernel = MlxMetalKernel::new(
        "prototype_jangtq2_prefill_gathered_dequant",
        &[
            "activations",
            "expert_assignment_starts",
            "packed_weights",
            "row_scales",
            "codebook_levels",
        ],
        &["projected"],
        PREFILL_GATHERED_DEQUANT_KERNEL_SOURCE,
    )
    .expect("the prototype kernel should compile");
    let expert_count = COMPLETE_LAYER_EXPERT_COUNT;
    let input_width = packed_words_per_row * u32::BITS as usize / bit_width as usize;

    let packed_words_host = deterministic_packed_words(
        u64::try_from(expert_count * 31 + packed_words_per_row).unwrap_or(0xA5A5),
        expert_count * output_rows * packed_words_per_row,
    );
    let row_scales_host = deterministic_row_scales(
        u64::try_from(expert_count * 17 + output_rows).unwrap_or(0x5A5A),
        expert_count * output_rows,
    );
    let packed_code_words = MlxArray::from_u32(
        &packed_words_host,
        &[
            expert_count as i32,
            output_rows as i32,
            packed_words_per_row as i32,
        ],
    )
    .expect("the synthetic packed page should form a valid uint32 tensor");
    let row_scales = runtime
        .astype(
            &runtime
                .array_from_f32(&row_scales_host, &[expert_count as i32, output_rows as i32])
                .expect("the synthetic row scales should upload"),
            MlxDtype::Float16,
        )
        .expect("the synthetic row scales should convert to float16");
    let codebook_levels_host =
        codebook::codebook_levels(bit_width).expect("the bit width is supported");
    let codebook_levels =
        MlxArray::from_f32(&codebook_levels_host, &[codebook_levels_host.len() as i32])
            .expect("the codebook level table should upload");
    codebook_levels
        .evaluate()
        .expect("the codebook level table should materialize");

    let assignments =
        expert_sorted_assignments(PREFILL_CHUNK_TOKENS, expert_count, TOP_K_EXPERT_COUNT);
    let assignment_count = assignments.assignment_expert_ids.len();

    // Token activations for the chunk, then per-assignment rows in expert-
    // sorted order (unrotated for the prototype, rotated for the baseline).
    let token_activation_host: Vec<f32> =
        deterministic_token_activations(0xC0FFEE, PREFILL_CHUNK_TOKENS * input_width);
    let token_activations = runtime
        .array_from_f32(
            &token_activation_host,
            &[PREFILL_CHUNK_TOKENS as i32, input_width as i32],
        )
        .expect("the chunk activations should upload");
    runtime
        .evaluate_arrays(&[&token_activations])
        .expect("the chunk activations should materialize");

    let mut unrotated_assignment_rows = vec![0.0f32; assignment_count * input_width];
    for (assignment_index, token_index) in assignments.assignment_token_indices.iter().enumerate() {
        unrotated_assignment_rows
            [assignment_index * input_width..(assignment_index + 1) * input_width]
            .copy_from_slice(
                &token_activation_host[token_index * input_width..(token_index + 1) * input_width],
            );
    }
    let prototype_assignments = runtime
        .array_from_f32(
            &unrotated_assignment_rows,
            &[assignment_count as i32, input_width as i32],
        )
        .expect("the prototype assignment rows should upload");
    runtime
        .evaluate_arrays(&[&prototype_assignments])
        .expect("the prototype assignment rows should materialize");

    // Baseline phase 1: the serving rotation graph on the chunk activations,
    // then on the per-assignment expansion the streamed path actually feeds
    // it (the diagnostic attribution showed both shapes in production).
    let mut rotation_chunk_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::rotate_activations_hadamard32(&runtime, &token_activations)
                .expect("the warmup chunk rotation should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let rotated = mlx_decode::rotate_activations_hadamard32(&runtime, &token_activations)
            .expect("the measured chunk rotation should succeed");
        rotation_chunk_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(rotated);
    }
    let rotated_chunk = mlx_decode::rotate_activations_hadamard32(&runtime, &token_activations)
        .expect("the reference chunk rotation should succeed");
    let rotated_chunk_host = runtime
        .array_to_vec_f32(&rotated_chunk)
        .expect("the rotated chunk should read back");
    let mut baseline_assignment_rows = vec![0.0f32; assignment_count * input_width];
    for (assignment_index, token_index) in assignments.assignment_token_indices.iter().enumerate() {
        baseline_assignment_rows
            [assignment_index * input_width..(assignment_index + 1) * input_width]
            .copy_from_slice(
                &rotated_chunk_host[token_index * input_width..(token_index + 1) * input_width],
            );
    }
    let baseline_assignments = runtime
        .array_from_f32(
            &baseline_assignment_rows,
            &[assignment_count as i32, input_width as i32],
        )
        .expect("the baseline assignment rows should upload");
    let baseline_assignments_expanded = runtime
        .reshape(
            &baseline_assignments,
            &[1, assignment_count as i32, 1, input_width as i32],
        )
        .expect("the baseline assignment rows should expand");
    runtime
        .evaluate_arrays(&[&baseline_assignments_expanded])
        .expect("the baseline assignment rows should materialize");
    let mut rotation_expanded_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::rotate_activations_hadamard32(&runtime, &baseline_assignments_expanded)
                .expect("the warmup expanded rotation should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let rotated =
            mlx_decode::rotate_activations_hadamard32(&runtime, &baseline_assignments_expanded)
                .expect("the measured expanded rotation should succeed");
        rotation_expanded_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(rotated);
    }

    // Baseline phase 2: the complete-layer page decode, already measured by
    // the page-decode lane; re-measured here so one run reports every phase.
    let mut decode_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, bit_width)
                .expect("the warmup page decode should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let decoded =
            decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, bit_width)
                .expect("the measured page decode should succeed");
        decode_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(decoded);
    }

    // Baseline phase 3: gather_mm over the decoded page, timed to a fully
    // evaluated output (the gathered matmul is lazy, so evaluation is the
    // real cost the serving chunk later waits on).
    let expert_assignment_starts = MlxArray::from_u32(
        &assignments.expert_assignment_starts,
        &[(expert_count + 1) as i32],
    )
    .expect("the compressed assignment starts should upload");
    let sorted_expert_ids = MlxArray::from_i32(
        &assignments
            .assignment_expert_ids
            .iter()
            .map(|&expert_index| expert_index as i32)
            .collect::<Vec<_>>(),
        &[1, assignment_count as i32],
    )
    .expect("the sorted expert ids should upload");
    // The decoded page is prepared once outside the timed window: the decode
    // stage is already timed separately, so embedding it here would
    // double-count it in the baseline sum below.
    let decoded_page =
        decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, bit_width)
            .expect("the gather-stage decode should succeed");
    let transposed_page = runtime
        .transpose_axes(&decoded_page, &[0, 2, 1])
        .expect("the gather-stage transpose should build");
    let mut gather_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut retained_gather_outputs = Vec::with_capacity(WARMUP_ITERATIONS + MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        let gathered = runtime
            .gather_dense_matmul(
                &baseline_assignments_expanded,
                &transposed_page,
                None,
                Some(&sorted_expert_ids),
                true,
            )
            .expect("the warmup gather should build");
        runtime
            .evaluate_arrays(&[&gathered])
            .expect("the warmup gather should evaluate");
        retained_gather_outputs.push(gathered);
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let gathered = runtime
            .gather_dense_matmul(
                &baseline_assignments_expanded,
                &transposed_page,
                None,
                Some(&sorted_expert_ids),
                true,
            )
            .expect("the measured gather should build");
        runtime
            .evaluate_arrays(&[&gathered])
            .expect("the measured gather should evaluate");
        gather_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        retained_gather_outputs.push(gathered);
    }

    // The prototype: one launch per projection, timed to an evaluated output.
    // Every output stays alive until after the parity read: a dropped output
    // freed from MLX's completion thread can have its buffer reused while a
    // later evaluation of an identical graph returns the stale node, which
    // measured impossibly fast medians (below the float32 arithmetic floor)
    // and zero parity rows before outputs were retained.
    let mut prototype_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut retained_prototype_outputs =
        Vec::with_capacity(WARMUP_ITERATIONS + MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        let projected = launch_prefill_gathered_dequant(
            &runtime,
            &kernel,
            &prototype_assignments,
            &expert_assignment_starts,
            &packed_code_words,
            &row_scales,
            &codebook_levels,
            bit_width,
            expert_count,
            output_rows,
        )
        .expect("the warmup prototype launch should succeed");
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the warmup prototype should evaluate");
        retained_prototype_outputs.push(projected);
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let projected = launch_prefill_gathered_dequant(
            &runtime,
            &kernel,
            &prototype_assignments,
            &expert_assignment_starts,
            &packed_code_words,
            &row_scales,
            &codebook_levels,
            bit_width,
            expert_count,
            output_rows,
        )
        .expect("the measured prototype launch should succeed");
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the measured prototype should evaluate");
        prototype_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        retained_prototype_outputs.push(projected);
    }

    // The gather comparison stays as reported context only: its bfloat16
    // weights and nondeterministic split-K ordering make it a performance
    // baseline, not a correctness oracle.
    let parity_assignments = runtime
        .array_from_f32(
            &unrotated_assignment_rows,
            &[assignment_count as i32, input_width as i32],
        )
        .expect("the parity assignment rows should upload freshly");
    let prototype_output = launch_prefill_gathered_dequant(
        &runtime,
        &kernel,
        &parity_assignments,
        &expert_assignment_starts,
        &packed_code_words,
        &row_scales,
        &codebook_levels,
        bit_width,
        expert_count,
        output_rows,
    )
    .expect("the parity prototype launch should succeed");
    runtime
        .evaluate_arrays(&[&prototype_output])
        .expect("the parity prototype should evaluate");
    let prototype_host = runtime
        .array_to_vec_f32(&prototype_output)
        .expect("the parity prototype output should read back");
    // Parity against exact host arithmetic, not against the gather baseline:
    // gather_mm's Metal split-K accumulates through atomic float adds, whose
    // order is nondeterministic run to run, so it can never pin a kernel.
    // The host reference evaluates the same math the kernel evaluates —
    // codebook levels times the blockwise Hadamard-32 of the activation row,
    // accumulated in float64 — for a strided sample of (assignment, row)
    // pairs small enough to stay cheap and large enough to cross threadgroup
    // boundaries.
    let sampled_assignments: Vec<usize> = (0..assignment_count).step_by(257).collect();
    let sampled_rows: Vec<usize> = (0..output_rows).step_by(97).collect();
    let mut maximum_exact_relative_error = 0.0f64;
    let mut exact_relative_error_sum = 0.0f64;
    let mut sampled_pair_count = 0_usize;
    for &assignment_index in &sampled_assignments {
        let expert_index = assignments.assignment_expert_ids[assignment_index] as usize;
        let rotated_row = super::super::hadamard32::reference_transform_vector(
            &unrotated_assignment_rows
                [assignment_index * input_width..(assignment_index + 1) * input_width],
        )
        .expect("the sampled assignment width is a whole number of blocks");
        for &row_index in &sampled_rows {
            let packed_row_offset = (expert_index * output_rows + row_index) * packed_words_per_row;
            let packed_row =
                &packed_words_host[packed_row_offset..packed_row_offset + packed_words_per_row];
            let codes = codebook::unpack_codes_from_packed_row(packed_row, bit_width)
                .expect("the sampled packed row is a valid code row");
            let row_scale = f64::from(row_scales_host[expert_index * output_rows + row_index]);
            let exact_dot: f64 = codes
                .iter()
                .zip(&rotated_row)
                .map(|(&code, &rotated_value)| {
                    f64::from(
                        codebook::codebook_level_value(bit_width, u32::from(code))
                            .expect("the sampled code is inside the codebook range"),
                    ) * f64::from(rotated_value)
                })
                .sum::<f64>()
                * row_scale;
            let prototype_value = prototype_host[assignment_index * output_rows + row_index];
            let denominator = exact_dot.abs().max(1.0);
            let relative_error = ((prototype_value as f64) - exact_dot).abs() / denominator;
            maximum_exact_relative_error = maximum_exact_relative_error.max(relative_error);
            exact_relative_error_sum += relative_error;
            sampled_pair_count += 1;
        }
    }
    let mean_exact_relative_error = exact_relative_error_sum / sampled_pair_count as f64;
    assert!(
        mean_exact_relative_error < 5e-4 && maximum_exact_relative_error < 2e-2,
        "case={case_name} prototype exact-parity mean relative error {mean_exact_relative_error:.6} (maximum {maximum_exact_relative_error:.5}) exceeds float32 accumulation rounding"
    );
    let decoded = decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, bit_width)
        .expect("the parity decode should succeed");
    let transposed = runtime
        .transpose_axes(&decoded, &[0, 2, 1])
        .expect("the parity transpose should build");
    let gathered = runtime
        .gather_dense_matmul(
            &baseline_assignments_expanded,
            &transposed,
            None,
            Some(&sorted_expert_ids),
            true,
        )
        .expect("the parity gather should build");
    runtime
        .evaluate_arrays(&[&gathered])
        .expect("the parity gather should evaluate");
    let gathered_host = runtime
        .array_to_vec_f32(
            &runtime
                .reshape(&gathered, &[assignment_count as i32, output_rows as i32])
                .expect("the parity gather output should flatten"),
        )
        .expect("the parity gather output should read back");
    assert_eq!(gathered_host.len(), prototype_host.len());
    let mut maximum_gather_relative_error = 0.0f64;
    let gather_relative_error_sum: f64 = prototype_host
        .iter()
        .zip(&gathered_host)
        .map(|(prototype_value, baseline_value)| {
            let denominator = f64::from(baseline_value.abs().max(1.0));
            let relative_error = ((prototype_value - baseline_value).abs()) as f64 / denominator;
            maximum_gather_relative_error = maximum_gather_relative_error.max(relative_error);
            relative_error
        })
        .sum();
    let mean_gather_relative_error = gather_relative_error_sum / prototype_host.len() as f64;
    let rotation_chunk_median_ms = median_duration_millis(&rotation_chunk_ms);
    let rotation_expanded_median_ms = median_duration_millis(&rotation_expanded_ms);
    let decode_median_ms = median_duration_millis(&decode_ms);
    let gather_median_ms = median_duration_millis(&gather_ms);
    let prototype_median_ms = median_duration_millis(&prototype_ms);
    let baseline_serving_median_ms =
        rotation_expanded_median_ms + decode_median_ms + gather_median_ms;
    let packed_gigabytes =
        (expert_count * output_rows * packed_words_per_row) as f64 * 4.0 / GIGABYTE_BYTES;
    let packed_gigabytes_per_second = packed_gigabytes / (prototype_median_ms / 1_000.0);
    eprintln!(
        "[jangtq2-perf:tier=prototype] case={case_name} tokens={PREFILL_CHUNK_TOKENS} assignments={assignment_count} bits={bit_width} rotation_chunk_ms={rotation_chunk_median_ms:.3} rotation_assignments_ms={rotation_expanded_median_ms:.3} page_decode_ms={decode_median_ms:.3} gather_ms={gather_median_ms:.3} baseline_serving_ms={baseline_serving_median_ms:.3} prototype_ms={prototype_median_ms:.3} speedup={:.2}x exact_parity_mean_rel_err={mean_exact_relative_error:.6} exact_parity_max_rel_err={maximum_exact_relative_error:.5} gather_parity_mean_rel_err={mean_gather_relative_error:.5} gather_parity_max_rel_err={maximum_gather_relative_error:.5} packed_gb_per_second={packed_gigabytes_per_second:.2}",
        baseline_serving_median_ms / prototype_median_ms,
    );
}

#[test]
#[serial]
#[ignore = "graphics-processor prototype measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_complete_layer_gate_up_prefill_gathered_dequant_throughput() {
    measure_prefill_gathered_dequant(
        "prefill_complete256_gate_up_2bit",
        GATE_UP_OUTPUT_ROWS,
        GATE_UP_PACKED_WORDS_2BIT,
        TWO_BIT_WIDTH,
    );
}

#[test]
#[serial]
#[ignore = "graphics-processor prototype measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_complete_layer_down_2bit_prefill_gathered_dequant_throughput() {
    measure_prefill_gathered_dequant(
        "prefill_complete256_down_2bit",
        DOWN_OUTPUT_ROWS,
        DOWN_PACKED_WORDS_2BIT,
        TWO_BIT_WIDTH,
    );
}

#[test]
#[serial]
#[ignore = "graphics-processor prototype measurement: run optimized via cargo test-jangtq2-decode-throughput"]
fn should_measure_complete_layer_down_3bit_prefill_gathered_dequant_throughput() {
    measure_prefill_gathered_dequant(
        "prefill_complete256_down_3bit",
        DOWN_OUTPUT_ROWS,
        DOWN_PACKED_WORDS_3BIT,
        THREE_BIT_WIDTH,
    );
}
