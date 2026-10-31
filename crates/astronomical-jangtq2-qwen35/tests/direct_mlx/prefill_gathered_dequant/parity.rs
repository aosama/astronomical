//! Parity verification for the prefill gathered-dequant prototype: the exact
//! host-arithmetic oracle that pins the kernel, and the gather baseline
//! comparison reported as context. Split from the measurement stages so each
//! concern stays small enough for the source-size budget.

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxMetalKernel};

use astronomical_jangtq2_qwen35::codebook;
use astronomical_jangtq2_qwen35::mlx_decode::decode_expert_weight_rows;

use super::kernel::{PrefillGatheredDequantInputs, launch_prefill_gathered_dequant};
use super::{COMPLETE_LAYER_EXPERT_COUNT, PrefillFixtureTensors, PrefillHostFixture};

/// Launches the prototype once more over a freshly uploaded copy of the
/// unrotated rows and reads the output back for the parity checks.
pub(super) fn read_exact_parity_output(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    fixture: &PrefillFixtureTensors,
    host_fixture: &PrefillHostFixture,
) -> Vec<f32> {
    let parity_assignments = runtime
        .array_from_f32(
            &host_fixture.unrotated_assignment_rows,
            &[
                host_fixture.assignment_count as i32,
                host_fixture.input_width as i32,
            ],
        )
        .expect("the parity assignment rows should upload freshly");
    let parity_inputs = PrefillGatheredDequantInputs {
        assignments: &parity_assignments,
        expert_assignment_starts: &fixture.expert_assignment_starts,
        packed_code_words: &fixture.packed_code_words,
        row_scales: &fixture.row_scales,
        codebook_level_table: &fixture.codebook_levels,
        bit_width: host_fixture.bit_width,
        expert_count: COMPLETE_LAYER_EXPERT_COUNT,
        output_rows: host_fixture.output_rows,
    };
    let prototype_output = launch_prefill_gathered_dequant(runtime, kernel, parity_inputs)
        .expect("the parity prototype launch should succeed");
    runtime
        .evaluate_arrays(&[&prototype_output])
        .expect("the parity prototype should evaluate");
    runtime
        .array_to_vec_f32(&prototype_output)
        .expect("the parity prototype output should read back")
}

/// Parity against exact host arithmetic, not against the gather baseline:
/// gather_mm's Metal split-K accumulates through atomic float adds, whose
/// order is nondeterministic run to run, so it can never pin a kernel. The
/// host reference evaluates the same math the kernel evaluates — codebook
/// levels times the blockwise Hadamard-32 of the activation row, accumulated
/// in float64 — for a strided sample of (assignment, row) pairs small enough
/// to stay cheap and large enough to cross threadgroup boundaries. Returns
/// the (mean, maximum) relative error after asserting the float32 budget.
pub(super) fn assert_exact_host_parity(
    prototype_host: &[f32],
    host_fixture: &PrefillHostFixture,
    case_name: &str,
) -> (f64, f64) {
    let input_width = host_fixture.input_width;
    let output_rows = host_fixture.output_rows;
    let packed_words_per_row = host_fixture.packed_words_per_row;
    let bit_width = host_fixture.bit_width;
    let assignment_count = host_fixture.assignment_count;
    let sampled_assignments: Vec<usize> = (0..assignment_count).step_by(257).collect();
    let sampled_rows: Vec<usize> = (0..output_rows).step_by(97).collect();
    let mut maximum_exact_relative_error = 0.0f64;
    let mut exact_relative_error_sum = 0.0f64;
    let mut sampled_pair_count = 0_usize;
    for &assignment_index in &sampled_assignments {
        let expert_index =
            host_fixture.assignments.assignment_expert_ids[assignment_index] as usize;
        let rotated_row = astronomical_jangtq2_qwen35::hadamard32::reference_transform_vector(
            &host_fixture.unrotated_assignment_rows
                [assignment_index * input_width..(assignment_index + 1) * input_width],
        )
        .expect("the sampled assignment width is a whole number of blocks");
        for &row_index in &sampled_rows {
            let packed_row_offset = (expert_index * output_rows + row_index) * packed_words_per_row;
            let packed_row = &host_fixture.packed_words_host
                [packed_row_offset..packed_row_offset + packed_words_per_row];
            let codes = codebook::unpack_codes_from_packed_row(packed_row, bit_width)
                .expect("the sampled packed row is a valid code row");
            let row_scale =
                f64::from(host_fixture.row_scales_host[expert_index * output_rows + row_index]);
            let mut exact_dot = 0.0f64;
            for (&code, &rotated_value) in codes.iter().zip(&rotated_row) {
                exact_dot += f64::from(
                    codebook::codebook_level_value(bit_width, u32::from(code))
                        .expect("the sampled code is inside the codebook range"),
                ) * f64::from(rotated_value);
            }
            let exact_dot = exact_dot * row_scale;
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
    (mean_exact_relative_error, maximum_exact_relative_error)
}

/// The gather comparison stays as reported context only: its bfloat16
/// weights and nondeterministic split-K ordering make it a performance
/// baseline, not a correctness oracle. Returns the (mean, maximum) relative
/// error of the prototype against that baseline.
pub(super) fn gather_baseline_parity_errors(
    runtime: &MlxRuntime,
    baseline_assignments_expanded: &MlxArray,
    fixture: &PrefillFixtureTensors,
    host_fixture: &PrefillHostFixture,
    prototype_host: &[f32],
) -> (f64, f64) {
    let decoded = decode_expert_weight_rows(
        runtime,
        &fixture.packed_code_words,
        &fixture.row_scales,
        host_fixture.bit_width,
    )
    .expect("the parity decode should succeed");
    let transposed = runtime
        .transpose_axes(&decoded, &[0, 2, 1])
        .expect("the parity transpose should build");
    let gathered = runtime
        .gather_dense_matmul(
            baseline_assignments_expanded,
            &transposed,
            None,
            Some(&fixture.sorted_expert_ids),
            true,
        )
        .expect("the parity gather should build");
    runtime
        .evaluate_arrays(&[&gathered])
        .expect("the parity gather should evaluate");
    let gathered_host = runtime
        .array_to_vec_f32(
            &runtime
                .reshape(
                    &gathered,
                    &[
                        host_fixture.assignment_count as i32,
                        host_fixture.output_rows as i32,
                    ],
                )
                .expect("the parity gather output should flatten"),
        )
        .expect("the parity gather output should read back");
    assert_eq!(gathered_host.len(), prototype_host.len());
    let mut maximum_gather_relative_error = 0.0f64;
    let mut gather_relative_error_sum = 0.0f64;
    for (prototype_value, baseline_value) in prototype_host.iter().zip(&gathered_host) {
        let denominator = f64::from(baseline_value.abs().max(1.0));
        let relative_error = ((prototype_value - baseline_value).abs()) as f64 / denominator;
        maximum_gather_relative_error = maximum_gather_relative_error.max(relative_error);
        gather_relative_error_sum += relative_error;
    }
    (
        gather_relative_error_sum / prototype_host.len() as f64,
        maximum_gather_relative_error,
    )
}

/// The four parity-error numbers the summary line reports, bundled so the
/// reporting stage takes one value instead of four loose arguments.
pub(super) struct ParityErrorStats {
    pub(super) exact_mean: f64,
    pub(super) exact_max: f64,
    pub(super) gather_mean: f64,
    pub(super) gather_max: f64,
}
