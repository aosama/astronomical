//! Correctness contracts for the fused jangtq2 gathered projection kernel.
//!
//! The host reference reproduces the format arithmetically — unpack the
//! LSB-first bitstream with `codebook::unpack_codes_from_packed_row`, evaluate
//! each level with `codebook::codebook_level_value`, multiply by the row scale,
//! and accumulate against the activation in float64 — so the kernel is pinned
//! to the format contract and not merely to the materializing decoder. A
//! second contract pins it to that decoder plus `gather_mm` on a real top-8
//! page shape, the pair it replaces on the decode path. Every test is
//! `#[serial]` because MLX graphics-processor tests never run in parallel.

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::super::codebook;
use super::super::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel;
use super::super::mlx_decode;

const TWO_BIT_WIDTH: u32 = 2;
const THREE_BIT_WIDTH: u32 = 3;
const FOUR_BIT_WIDTH: u32 = 4;
/// Real top-8 gate/up page geometry of the 122B bundle.
const TOP_K_EXPERT_COUNT: usize = 8;
const GATE_UP_OUTPUT_ROWS: usize = 1024;
const GATE_UP_INPUT_WIDTH: usize = 3072;
/// Float32 accumulation over a few hundred terms against a float64 reference.
const FLOAT32_RELATIVE_TOLERANCE: f64 = 1e-4;
/// bfloat16 weights in the materializing path carry ~3 significant digits;
/// the fused path keeps float32 levels, so the two differ at that scale.
const BFLOAT16_RELATIVE_TOLERANCE: f64 = 2.5e-2;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the fused projection test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Deterministic splitmix64 words so every run multiplies the same codes.
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

/// Row scales exactly representable in float16.
fn deterministic_row_scales(seed: u64, row_count: usize) -> Vec<f32> {
    (0..row_count)
        .map(|row_index| ((seed.wrapping_add(row_index as u64) % 16 + 4) as f32) / 8.0)
        .collect()
}

/// Activation values exactly representable in bfloat16 so the bfloat16
/// contract compares the same inputs on both paths.
fn deterministic_activations(seed: u64, value_count: usize) -> Vec<f32> {
    (0..value_count)
        .map(|value_index| {
            let deterministic_step = (seed.wrapping_add(value_index as u64 * 7) % 33) as f32;
            (deterministic_step - 16.0) / 16.0
        })
        .collect()
}

struct FusedProjectionFixture {
    output_rows: usize,
    input_width: usize,
    bit_width: u32,
    packed_words_host: Vec<u32>,
    row_scales_host: Vec<f32>,
    packed_code_words: MlxArray,
    row_scales: MlxArray,
}

fn build_fixture(
    runtime: &MlxRuntime,
    expert_count: usize,
    output_rows: usize,
    input_width: usize,
    bit_width: u32,
) -> FusedProjectionFixture {
    let words_per_row = input_width * bit_width as usize / 32;
    let packed_words_host = deterministic_packed_words(
        (expert_count * 131 + input_width) as u64,
        expert_count * output_rows * words_per_row,
    );
    let row_scales_host =
        deterministic_row_scales((output_rows * 17) as u64, expert_count * output_rows);
    let packed_code_words = MlxArray::from_u32(
        &packed_words_host,
        &[
            expert_count as i32,
            output_rows as i32,
            words_per_row as i32,
        ],
    )
    .expect("the packed fixture should form a uint32 tensor");
    let row_scales_float32 = runtime
        .array_from_f32(&row_scales_host, &[expert_count as i32, output_rows as i32])
        .expect("the row scale fixture should upload");
    let row_scales = runtime
        .astype(&row_scales_float32, MlxDtype::Float16)
        .expect("the row scales should convert to float16");
    runtime
        .evaluate_arrays(&[&packed_code_words, &row_scales])
        .expect("the fixture should materialize");
    FusedProjectionFixture {
        output_rows,
        input_width,
        bit_width,
        packed_words_host,
        row_scales_host,
        packed_code_words,
        row_scales,
    }
}

impl FusedProjectionFixture {
    /// `scale[e, r] · Σ_j level(code[e, r, j]) · x[j]` in float64.
    fn reference_projection(&self, hidden_rows: &[Vec<f32>], expert_indices: &[i32]) -> Vec<f64> {
        let words_per_row = self.input_width * self.bit_width as usize / 32;
        let mut projected = Vec::with_capacity(expert_indices.len() * self.output_rows);
        for (assignment_index, &expert_index) in expert_indices.iter().enumerate() {
            let hidden_row = if hidden_rows.len() == 1 {
                &hidden_rows[0]
            } else {
                &hidden_rows[assignment_index]
            };
            let expert_index = expert_index as usize;
            for output_row in 0..self.output_rows {
                let row_start = (expert_index * self.output_rows + output_row) * words_per_row;
                let codes = codebook::unpack_codes_from_packed_row(
                    &self.packed_words_host[row_start..row_start + words_per_row],
                    self.bit_width,
                )
                .expect("fixture rows are whole 32-code groups");
                let dot: f64 = codes
                    .iter()
                    .zip(hidden_row)
                    .map(|(&code, &activation)| {
                        f64::from(
                            codebook::codebook_level_value(self.bit_width, u32::from(code))
                                .expect("fixture codes are in range"),
                        ) * f64::from(activation)
                    })
                    .sum();
                let scale =
                    f64::from(self.row_scales_host[expert_index * self.output_rows + output_row]);
                projected.push(dot * scale);
            }
        }
        projected
    }
}

fn assert_close(actual: &[f32], expected: &[f64], relative_tolerance: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: element count");
    let largest_expected_magnitude = expected
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
        .max(1e-6);
    for (element_index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate()
    {
        let difference = (f64::from(actual_value) - expected_value).abs();
        assert!(
            difference <= relative_tolerance * largest_expected_magnitude,
            "{context}: element {element_index} fused={actual_value} reference={expected_value} (tolerance {relative_tolerance} × {largest_expected_magnitude})"
        );
    }
    assert!(
        actual.iter().any(|value| *value != 0.0),
        "{context}: an all-zero projection is the signature of a dropped dispatch"
    );
}

fn run_host_reference_contract(
    bit_width: u32,
    input_width: usize,
    expert_indices_host: &[i32],
    per_assignment_rows: bool,
    context: &str,
) {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(&runtime, 5, 64, input_width, bit_width);
    let hidden_row_count = if per_assignment_rows {
        expert_indices_host.len()
    } else {
        1
    };
    let hidden_rows: Vec<Vec<f32>> = (0..hidden_row_count)
        .map(|row_index| deterministic_activations(row_index as u64 + 11, input_width))
        .collect();
    let hidden = runtime
        .array_from_f32(
            &hidden_rows.concat(),
            &[hidden_row_count as i32, input_width as i32],
        )
        .expect("the activation fixture should upload");
    let expert_indices =
        MlxArray::from_i32(expert_indices_host, &[expert_indices_host.len() as i32])
            .expect("the expert index fixture should upload");
    let projected = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            bit_width,
        )
        .expect("the fused projection should launch");
    assert_eq!(
        projected.shape(),
        [expert_indices_host.len() as i32, fixture.output_rows as i32]
    );
    assert_eq!(projected.dtype(), MlxDtype::Float32);
    let projected_host = runtime
        .array_to_vec_f32(&projected)
        .expect("the projection should read back");
    let expected = fixture.reference_projection(&hidden_rows, expert_indices_host);
    assert_close(
        &projected_host,
        &expected,
        FLOAT32_RELATIVE_TOLERANCE,
        context,
    );
}

/// Duplicate and out-of-order experts: the kernel must not assume sorted or
/// unique indices.
const UNSORTED_DUPLICATE_ROUTE: &[i32] = &[3, 0, 4, 3];

#[test]
#[serial]
fn should_match_host_reference_for_two_bit_codes_with_one_shared_activation_row() {
    run_host_reference_contract(
        TWO_BIT_WIDTH,
        96,
        UNSORTED_DUPLICATE_ROUTE,
        false,
        "two-bit shared row",
    );
}

#[test]
#[serial]
fn should_match_host_reference_for_three_bit_codes_that_straddle_words() {
    // At 3 bits every 32-code group spans three words and ten codes straddle
    // a word boundary, the layout the kernel's spread-and-mask must honor.
    run_host_reference_contract(
        THREE_BIT_WIDTH,
        96,
        UNSORTED_DUPLICATE_ROUTE,
        false,
        "three-bit straddling codes",
    );
}

#[test]
#[serial]
fn should_match_host_reference_for_four_bit_codes() {
    run_host_reference_contract(
        FOUR_BIT_WIDTH,
        64,
        UNSORTED_DUPLICATE_ROUTE,
        false,
        "four-bit codes",
    );
}

#[test]
#[serial]
fn should_match_host_reference_with_one_activation_row_per_assignment() {
    run_host_reference_contract(
        TWO_BIT_WIDTH,
        96,
        UNSORTED_DUPLICATE_ROUTE,
        true,
        "per-assignment rows",
    );
}

#[test]
#[serial]
fn should_project_a_single_assignment_route() {
    // The one-assignment boundary: the grid is exactly one expert's rows and
    // the shared-row and per-assignment layouts coincide.
    run_host_reference_contract(TWO_BIT_WIDTH, 64, &[4], false, "single assignment");
}

#[test]
#[serial]
fn should_match_materializing_decode_and_gather_matmul_on_a_top_k_page() {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(
        &runtime,
        TOP_K_EXPERT_COUNT,
        GATE_UP_OUTPUT_ROWS,
        GATE_UP_INPUT_WIDTH,
        TWO_BIT_WIDTH,
    );
    let expert_indices_host: Vec<i32> = (0..TOP_K_EXPERT_COUNT as i32).rev().collect();
    let activations_host = deterministic_activations(29, GATE_UP_INPUT_WIDTH);
    let hidden_float32 = runtime
        .array_from_f32(&activations_host, &[1, GATE_UP_INPUT_WIDTH as i32])
        .expect("the activation fixture should upload");
    let hidden = runtime
        .astype(&hidden_float32, MlxDtype::BFloat16)
        .expect("the activations should convert to bfloat16");

    // The materializing path exactly as `paged_execution` runs it: decode the
    // page to bfloat16, transpose lazily, gather_mm with [1, 1, 1, H] inputs.
    let decoded_page = mlx_decode::decode_expert_weight_rows(
        &runtime,
        &fixture.packed_code_words,
        &fixture.row_scales,
        TWO_BIT_WIDTH,
    )
    .expect("the materializing decode should succeed");
    let transposed_page = runtime
        .transpose_axes(&decoded_page, &[0, 2, 1])
        .expect("the decoded page should transpose");
    let expanded_hidden = runtime
        .reshape(&hidden, &[1, 1, 1, GATE_UP_INPUT_WIDTH as i32])
        .expect("the activations should expand");
    let route_indices = MlxArray::from_i32(&expert_indices_host, &[1, TOP_K_EXPERT_COUNT as i32])
        .expect("the route fixture should upload");
    let gathered = runtime
        .gather_dense_matmul(
            &expanded_hidden,
            &transposed_page,
            None,
            Some(&route_indices),
            false,
        )
        .expect("gather_mm should build");
    let gathered_host = runtime
        .array_to_vec_f32(
            &runtime
                .astype(&gathered, MlxDtype::Float32)
                .expect("the gathered output should widen"),
        )
        .expect("the gathered output should read back");

    let expert_indices = MlxArray::from_i32(&expert_indices_host, &[TOP_K_EXPERT_COUNT as i32])
        .expect("the flat route fixture should upload");
    let projected = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            TWO_BIT_WIDTH,
        )
        .expect("the fused projection should launch");
    assert_eq!(projected.dtype(), MlxDtype::BFloat16);
    let projected_host = runtime
        .array_to_vec_f32(
            &runtime
                .astype(&projected, MlxDtype::Float32)
                .expect("the projection should widen"),
        )
        .expect("the projection should read back");

    let expected: Vec<f64> = gathered_host
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    assert_close(
        &projected_host,
        &expected,
        BFLOAT16_RELATIVE_TOLERANCE,
        "fused versus materializing decode plus gather_mm",
    );
    // Both paths must also agree with the format's own arithmetic.
    let reference = fixture.reference_projection(&[activations_host], &expert_indices_host);
    assert_close(
        &projected_host,
        &reference,
        BFLOAT16_RELATIVE_TOLERANCE,
        "fused versus host reference on the top-8 page",
    );
}

#[test]
#[serial]
fn should_reject_an_unsupported_bit_width() {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(&runtime, 2, 32, 64, TWO_BIT_WIDTH);
    let hidden = runtime
        .array_from_f32(&deterministic_activations(1, 64), &[1, 64])
        .expect("the activation fixture should upload");
    let expert_indices = MlxArray::from_i32(&[0], &[1]).expect("the route fixture should upload");
    let error = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            8,
        )
        .expect_err("an 8-bit width is outside the codebook family");
    assert!(
        error
            .to_string()
            .contains("unsupported jangtq2 codebook bit width")
    );
}

#[test]
#[serial]
fn should_reject_activations_whose_width_disagrees_with_the_packed_rows() {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(&runtime, 2, 32, 64, TWO_BIT_WIDTH);
    let hidden = runtime
        .array_from_f32(&deterministic_activations(1, 96), &[1, 96])
        .expect("the activation fixture should upload");
    let expert_indices = MlxArray::from_i32(&[1], &[1]).expect("the route fixture should upload");
    let error = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            TWO_BIT_WIDTH,
        )
        .expect_err("a 96-wide activation cannot contract against 64-wide rows");
    assert!(error.to_string().contains("activations must be"));
}

#[test]
#[serial]
fn should_reject_activation_row_counts_that_match_neither_one_nor_the_assignment_count() {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(&runtime, 2, 32, 64, TWO_BIT_WIDTH);
    let hidden = runtime
        .array_from_f32(&deterministic_activations(1, 128), &[2, 64])
        .expect("the activation fixture should upload");
    let expert_indices =
        MlxArray::from_i32(&[1, 0, 1], &[3]).expect("the route fixture should upload");
    let error = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            TWO_BIT_WIDTH,
        )
        .expect_err("two activation rows cannot serve three assignments");
    assert!(error.to_string().contains("assignments"));
}

#[test]
#[serial]
fn should_reject_floating_point_expert_indices() {
    let runtime = test_runtime();
    let kernel = FusedJangtq2GatheredProjectionKernel::new()
        .expect("the fused projection kernel should compile");
    let fixture = build_fixture(&runtime, 2, 32, 64, TWO_BIT_WIDTH);
    let hidden = runtime
        .array_from_f32(&deterministic_activations(1, 64), &[1, 64])
        .expect("the activation fixture should upload");
    let expert_indices = runtime
        .array_from_f32(&[1.0], &[1])
        .expect("the route fixture should upload");
    let error = kernel
        .apply(
            &runtime,
            &hidden,
            &expert_indices,
            &fixture.packed_code_words,
            &fixture.row_scales,
            TWO_BIT_WIDTH,
        )
        .expect_err("float indices cannot address experts");
    assert!(
        error
            .to_string()
            .contains("expert indices must use int32 or uint32")
    );
}
