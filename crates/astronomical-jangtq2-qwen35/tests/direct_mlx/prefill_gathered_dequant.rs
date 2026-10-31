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
//! Every test is `#[serial]` because these are throughput measurements,
//! and they are `#[ignore]`d: run them optimized through
//! `cargo test-jangtq2-decode-throughput` (the filter matches this module's
//! `..._throughput` names).

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype, MlxMetalKernel};
use serial_test::serial;

use astronomical_jangtq2_qwen35::codebook;
use astronomical_jangtq2_qwen35::mlx_decode;
use astronomical_jangtq2_qwen35::mlx_decode::decode_expert_weight_rows;
use fixtures::{
    deterministic_packed_words, deterministic_row_scales, deterministic_token_activations,
    expert_sorted_assignments, median_duration_millis, prototype_runtime,
};
use kernel::{
    PREFILL_GATHERED_DEQUANT_KERNEL_SOURCE, PrefillGatheredDequantInputs,
    launch_prefill_gathered_dequant,
};

mod fixtures;
mod kernel;
mod parity;

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
///
/// The orchestrator reads as the experiment's protocol; each stage lives in
/// its own function below. Every timed stage keeps its own warmup loop
/// because a shared timing harness would need closures, which the repository
/// forbids. Retained outputs are returned to the caller so the buffers the
/// measurement kept alive stay alive for exactly as long as the original
/// single function held them.
fn measure_prefill_gathered_dequant(
    case_name: &str,
    output_rows: usize,
    packed_words_per_row: usize,
    bit_width: u32,
) {
    let runtime = prototype_runtime();
    let kernel = compile_prefill_prototype_kernel();
    let expert_count = COMPLETE_LAYER_EXPERT_COUNT;
    let host_fixture =
        build_prefill_host_fixture(expert_count, output_rows, packed_words_per_row, bit_width);
    let fixture = upload_prefill_fixture(&runtime, &host_fixture);

    let (rotation_chunk_ms, rotated_chunk_host) =
        time_chunk_rotation(&runtime, &fixture.token_activations);
    let baseline_assignments_expanded =
        upload_expanded_baseline_assignments(&runtime, &rotated_chunk_host, &host_fixture);
    let rotation_expanded_ms = time_expanded_rotation(&runtime, &baseline_assignments_expanded);
    let decode_ms = time_page_decode(
        &runtime,
        &fixture.packed_code_words,
        &fixture.row_scales,
        bit_width,
    );
    let transposed_page = prepare_transposed_decoded_page(
        &runtime,
        &fixture.packed_code_words,
        &fixture.row_scales,
        bit_width,
    );
    let (gather_ms, _retained_gather_outputs) = time_gather_matmul(
        &runtime,
        &baseline_assignments_expanded,
        &transposed_page,
        &fixture.sorted_expert_ids,
    );
    let (prototype_ms, _retained_prototype_outputs) = time_prototype_kernel(
        &runtime,
        &kernel,
        &fixture,
        bit_width,
        expert_count,
        output_rows,
    );
    let prototype_host =
        parity::read_exact_parity_output(&runtime, &kernel, &fixture, &host_fixture);
    let (mean_exact_relative_error, maximum_exact_relative_error) =
        parity::assert_exact_host_parity(&prototype_host, &host_fixture, case_name);
    let (mean_gather_relative_error, maximum_gather_relative_error) =
        parity::gather_baseline_parity_errors(
            &runtime,
            &baseline_assignments_expanded,
            &fixture,
            &host_fixture,
            &prototype_host,
        );
    let phase_samples = PrefillPhaseSamples {
        rotation_chunk_ms: &rotation_chunk_ms,
        rotation_expanded_ms: &rotation_expanded_ms,
        decode_ms: &decode_ms,
        gather_ms: &gather_ms,
        prototype_ms: &prototype_ms,
    };
    let parity_errors = parity::ParityErrorStats {
        exact_mean: mean_exact_relative_error,
        exact_max: maximum_exact_relative_error,
        gather_mean: mean_gather_relative_error,
        gather_max: maximum_gather_relative_error,
    };
    report_prefill_measurement(
        case_name,
        &host_fixture,
        expert_count,
        &phase_samples,
        &parity_errors,
    );
}

/// Compiles the prototype Metal kernel once per measurement run.
fn compile_prefill_prototype_kernel() -> MlxMetalKernel {
    MlxMetalKernel::new(
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
    .expect("the prototype kernel should compile")
}

/// Host-side deterministic inputs for one measurement case, plus the derived
/// shapes every stage shares. Built once so each stage reads identical data.
struct PrefillHostFixture {
    packed_words_host: Vec<u32>,
    row_scales_host: Vec<f32>,
    token_activation_host: Vec<f32>,
    unrotated_assignment_rows: Vec<f32>,
    assignments: fixtures::SortedAssignments,
    assignment_count: usize,
    input_width: usize,
    packed_words_per_row: usize,
    bit_width: u32,
    output_rows: usize,
}

fn build_prefill_host_fixture(
    expert_count: usize,
    output_rows: usize,
    packed_words_per_row: usize,
    bit_width: u32,
) -> PrefillHostFixture {
    let input_width = packed_words_per_row * u32::BITS as usize / bit_width as usize;
    let packed_words_host = deterministic_packed_words(
        u64::try_from(expert_count * 31 + packed_words_per_row).unwrap_or(0xA5A5),
        expert_count * output_rows * packed_words_per_row,
    );
    let row_scales_host = deterministic_row_scales(
        u64::try_from(expert_count * 17 + output_rows).unwrap_or(0x5A5A),
        expert_count * output_rows,
    );
    let assignments =
        expert_sorted_assignments(PREFILL_CHUNK_TOKENS, expert_count, TOP_K_EXPERT_COUNT);
    let assignment_count = assignments.assignment_expert_ids.len();
    // Token activations for the chunk, then per-assignment rows in expert-
    // sorted order (unrotated for the prototype, rotated for the baseline).
    let token_activation_host: Vec<f32> =
        deterministic_token_activations(0xC0FFEE, PREFILL_CHUNK_TOKENS * input_width);
    let unrotated_assignment_rows =
        gather_assignment_rows(&token_activation_host, &assignments, input_width);

    PrefillHostFixture {
        packed_words_host,
        row_scales_host,
        token_activation_host,
        unrotated_assignment_rows,
        assignments,
        assignment_count,
        input_width,
        packed_words_per_row,
        bit_width,
        output_rows,
    }
}

/// Expands per-token rows into expert-sorted per-assignment rows: the
/// prototype feeds unrotated rows this way and the baseline feeds rotated
/// rows this way, so both expansions share one implementation.
fn gather_assignment_rows(
    source_rows: &[f32],
    assignments: &fixtures::SortedAssignments,
    input_width: usize,
) -> Vec<f32> {
    let mut assignment_rows =
        vec![0.0f32; assignments.assignment_token_indices.len() * input_width];
    for (assignment_index, token_index) in assignments.assignment_token_indices.iter().enumerate() {
        assignment_rows[assignment_index * input_width..(assignment_index + 1) * input_width]
            .copy_from_slice(
                &source_rows[token_index * input_width..(token_index + 1) * input_width],
            );
    }
    assignment_rows
}

/// Device-resident tensors every stage consumes. Uploaded once, evaluated
/// eagerly, then shared by reference for the rest of the run.
struct PrefillFixtureTensors {
    packed_code_words: MlxArray,
    row_scales: MlxArray,
    codebook_levels: MlxArray,
    token_activations: MlxArray,
    prototype_assignments: MlxArray,
    expert_assignment_starts: MlxArray,
    sorted_expert_ids: MlxArray,
}

fn upload_prefill_fixture(
    runtime: &MlxRuntime,
    host_fixture: &PrefillHostFixture,
) -> PrefillFixtureTensors {
    let expert_count = COMPLETE_LAYER_EXPERT_COUNT;
    let packed_code_words = MlxArray::from_u32(
        &host_fixture.packed_words_host,
        &[
            expert_count as i32,
            host_fixture.output_rows as i32,
            host_fixture.packed_words_per_row as i32,
        ],
    )
    .expect("the synthetic packed page should form a valid uint32 tensor");
    let row_scales = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &host_fixture.row_scales_host,
                    &[expert_count as i32, host_fixture.output_rows as i32],
                )
                .expect("the synthetic row scales should upload"),
            MlxDtype::Float16,
        )
        .expect("the synthetic row scales should convert to float16");
    let codebook_levels_host =
        codebook::codebook_levels(host_fixture.bit_width).expect("the bit width is supported");
    let codebook_levels =
        MlxArray::from_f32(&codebook_levels_host, &[codebook_levels_host.len() as i32])
            .expect("the codebook level table should upload");
    codebook_levels
        .evaluate()
        .expect("the codebook level table should materialize");
    let token_activations = runtime
        .array_from_f32(
            &host_fixture.token_activation_host,
            &[PREFILL_CHUNK_TOKENS as i32, host_fixture.input_width as i32],
        )
        .expect("the chunk activations should upload");
    runtime
        .evaluate_arrays(&[&token_activations])
        .expect("the chunk activations should materialize");
    let prototype_assignments = runtime
        .array_from_f32(
            &host_fixture.unrotated_assignment_rows,
            &[
                host_fixture.assignment_count as i32,
                host_fixture.input_width as i32,
            ],
        )
        .expect("the prototype assignment rows should upload");
    runtime
        .evaluate_arrays(&[&prototype_assignments])
        .expect("the prototype assignment rows should materialize");
    let expert_assignment_starts = MlxArray::from_u32(
        &host_fixture.assignments.expert_assignment_starts,
        &[(expert_count + 1) as i32],
    )
    .expect("the compressed assignment starts should upload");
    let mut sorted_expert_ids_host = Vec::with_capacity(host_fixture.assignment_count);
    for &expert_index in &host_fixture.assignments.assignment_expert_ids {
        sorted_expert_ids_host.push(expert_index as i32);
    }
    let sorted_expert_ids = MlxArray::from_i32(
        &sorted_expert_ids_host,
        &[1, host_fixture.assignment_count as i32],
    )
    .expect("the sorted expert ids should upload");

    PrefillFixtureTensors {
        packed_code_words,
        row_scales,
        codebook_levels,
        token_activations,
        prototype_assignments,
        expert_assignment_starts,
        sorted_expert_ids,
    }
}

/// Baseline phase 1a: times the serving rotation graph on the chunk
/// activations (the diagnostic attribution showed this shape in production),
/// then returns the reference rotation's host copy for the baseline expansion.
fn time_chunk_rotation(runtime: &MlxRuntime, token_activations: &MlxArray) -> (Vec<f64>, Vec<f32>) {
    let mut rotation_chunk_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::rotate_activations_hadamard32(runtime, token_activations)
                .expect("the warmup chunk rotation should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let rotated = mlx_decode::rotate_activations_hadamard32(runtime, token_activations)
            .expect("the measured chunk rotation should succeed");
        rotation_chunk_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(rotated);
    }
    let rotated_chunk = mlx_decode::rotate_activations_hadamard32(runtime, token_activations)
        .expect("the reference chunk rotation should succeed");
    let rotated_chunk_host = runtime
        .array_to_vec_f32(&rotated_chunk)
        .expect("the rotated chunk should read back");
    (rotation_chunk_ms, rotated_chunk_host)
}

/// Baseline phase 1b: expands the rotated chunk into per-assignment rows and
/// uploads them in the [1, assignments, 1, width] shape the streamed path
/// actually feeds `gather_mm`.
fn upload_expanded_baseline_assignments(
    runtime: &MlxRuntime,
    rotated_chunk_host: &[f32],
    host_fixture: &PrefillHostFixture,
) -> MlxArray {
    let baseline_assignment_rows = gather_assignment_rows(
        rotated_chunk_host,
        &host_fixture.assignments,
        host_fixture.input_width,
    );
    let baseline_assignments = runtime
        .array_from_f32(
            &baseline_assignment_rows,
            &[
                host_fixture.assignment_count as i32,
                host_fixture.input_width as i32,
            ],
        )
        .expect("the baseline assignment rows should upload");
    let baseline_assignments_expanded = runtime
        .reshape(
            &baseline_assignments,
            &[
                1,
                host_fixture.assignment_count as i32,
                1,
                host_fixture.input_width as i32,
            ],
        )
        .expect("the baseline assignment rows should expand");
    runtime
        .evaluate_arrays(&[&baseline_assignments_expanded])
        .expect("the baseline assignment rows should materialize");
    baseline_assignments_expanded
}

/// Baseline phase 1c: times the rotation graph on the expanded
/// per-assignment rows.
fn time_expanded_rotation(runtime: &MlxRuntime, expanded: &MlxArray) -> Vec<f64> {
    let mut rotation_expanded_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            mlx_decode::rotate_activations_hadamard32(runtime, expanded)
                .expect("the warmup expanded rotation should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let rotated = mlx_decode::rotate_activations_hadamard32(runtime, expanded)
            .expect("the measured expanded rotation should succeed");
        rotation_expanded_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(rotated);
    }
    rotation_expanded_ms
}

/// Baseline phase 2: the complete-layer page decode, already measured by the
/// page-decode lane; re-measured here so one run reports every phase.
fn time_page_decode(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) -> Vec<f64> {
    let mut decode_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        drop(
            decode_expert_weight_rows(runtime, packed_code_words, row_scales, bit_width)
                .expect("the warmup page decode should succeed"),
        );
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let decoded = decode_expert_weight_rows(runtime, packed_code_words, row_scales, bit_width)
            .expect("the measured page decode should succeed");
        decode_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        drop(decoded);
    }
    decode_ms
}

/// Prepares the decoded page for the gather stage. Prepared once outside the
/// timed window: the decode stage is already timed separately, so embedding
/// it in the gather stage would double-count it in the baseline sum.
fn prepare_transposed_decoded_page(
    runtime: &MlxRuntime,
    packed_code_words: &MlxArray,
    row_scales: &MlxArray,
    bit_width: u32,
) -> MlxArray {
    let decoded_page = decode_expert_weight_rows(runtime, packed_code_words, row_scales, bit_width)
        .expect("the gather-stage decode should succeed");
    runtime
        .transpose_axes(&decoded_page, &[0, 2, 1])
        .expect("the gather-stage transpose should build")
}

/// Baseline phase 3: `gather_mm` over the decoded page, timed to a fully
/// evaluated output (the gathered matmul is lazy, so evaluation is the real
/// cost the serving chunk later waits on). Every timed output stays alive
/// until the caller drops the returned vector.
fn time_gather_matmul(
    runtime: &MlxRuntime,
    expanded: &MlxArray,
    transposed_page: &MlxArray,
    sorted_expert_ids: &MlxArray,
) -> (Vec<f64>, Vec<MlxArray>) {
    let mut gather_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut retained_gather_outputs = Vec::with_capacity(WARMUP_ITERATIONS + MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        let gathered = runtime
            .gather_dense_matmul(
                expanded,
                transposed_page,
                None,
                Some(sorted_expert_ids),
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
                expanded,
                transposed_page,
                None,
                Some(sorted_expert_ids),
                true,
            )
            .expect("the measured gather should build");
        runtime
            .evaluate_arrays(&[&gathered])
            .expect("the measured gather should evaluate");
        gather_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        retained_gather_outputs.push(gathered);
    }
    (gather_ms, retained_gather_outputs)
}

/// The prototype: one launch per projection, timed to an evaluated output.
/// Every output stays alive until the caller drops the returned vector: a
/// dropped output freed from MLX's completion thread can have its buffer
/// reused while a later evaluation of an identical graph returns the stale
/// node, which measured impossibly fast medians (below the float32
/// arithmetic floor) and zero parity rows before outputs were retained.
fn time_prototype_kernel(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    fixture: &PrefillFixtureTensors,
    bit_width: u32,
    expert_count: usize,
    output_rows: usize,
) -> (Vec<f64>, Vec<MlxArray>) {
    let prototype_inputs = PrefillGatheredDequantInputs {
        assignments: &fixture.prototype_assignments,
        expert_assignment_starts: &fixture.expert_assignment_starts,
        packed_code_words: &fixture.packed_code_words,
        row_scales: &fixture.row_scales,
        codebook_level_table: &fixture.codebook_levels,
        bit_width,
        expert_count,
        output_rows,
    };
    let mut prototype_ms = Vec::with_capacity(MEASURED_ITERATIONS);
    let mut retained_prototype_outputs =
        Vec::with_capacity(WARMUP_ITERATIONS + MEASURED_ITERATIONS);
    for _ in 0..WARMUP_ITERATIONS {
        let projected = launch_prefill_gathered_dequant(runtime, kernel, prototype_inputs)
            .expect("the warmup prototype launch should succeed");
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the warmup prototype should evaluate");
        retained_prototype_outputs.push(projected);
    }
    for _ in 0..MEASURED_ITERATIONS {
        let started_at = std::time::Instant::now();
        let projected = launch_prefill_gathered_dequant(runtime, kernel, prototype_inputs)
            .expect("the measured prototype launch should succeed");
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the measured prototype should evaluate");
        prototype_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        retained_prototype_outputs.push(projected);
    }
    (prototype_ms, retained_prototype_outputs)
}

/// The per-phase timing samples, bundled so the reporting stage takes one
/// value instead of five loose slice arguments.
struct PrefillPhaseSamples<'a> {
    rotation_chunk_ms: &'a [f64],
    rotation_expanded_ms: &'a [f64],
    decode_ms: &'a [f64],
    gather_ms: &'a [f64],
    prototype_ms: &'a [f64],
}

/// Reduces the per-phase samples to medians and prints the one-line summary
/// the throughput lanes parse.
fn report_prefill_measurement(
    case_name: &str,
    host_fixture: &PrefillHostFixture,
    expert_count: usize,
    phase_samples: &PrefillPhaseSamples<'_>,
    parity_errors: &parity::ParityErrorStats,
) {
    let bit_width = host_fixture.bit_width;
    let output_rows = host_fixture.output_rows;
    let packed_words_per_row = host_fixture.packed_words_per_row;
    let rotation_chunk_ms = phase_samples.rotation_chunk_ms;
    let rotation_expanded_ms = phase_samples.rotation_expanded_ms;
    let decode_ms = phase_samples.decode_ms;
    let gather_ms = phase_samples.gather_ms;
    let prototype_ms = phase_samples.prototype_ms;
    let mean_exact_relative_error = parity_errors.exact_mean;
    let maximum_exact_relative_error = parity_errors.exact_max;
    let mean_gather_relative_error = parity_errors.gather_mean;
    let maximum_gather_relative_error = parity_errors.gather_max;
    let rotation_chunk_median_ms = median_duration_millis(rotation_chunk_ms);
    let rotation_expanded_median_ms = median_duration_millis(rotation_expanded_ms);
    let decode_median_ms = median_duration_millis(decode_ms);
    let gather_median_ms = median_duration_millis(gather_ms);
    let prototype_median_ms = median_duration_millis(prototype_ms);
    let baseline_serving_median_ms =
        rotation_expanded_median_ms + decode_median_ms + gather_median_ms;
    let packed_gigabytes =
        (expert_count * output_rows * packed_words_per_row) as f64 * 4.0 / GIGABYTE_BYTES;
    let packed_gigabytes_per_second = packed_gigabytes / (prototype_median_ms / 1_000.0);
    let assignment_count = host_fixture.assignment_count;
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
