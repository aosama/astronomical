//! Correctness contracts for the jangtq2 graphics-processor codebook decoder
//! and the blockwise Hadamard-32 activation rotation in `mlx_decode`, pinned
//! to the checked-in golden vectors and to the format's host arithmetic.
//! Every test is `#[serial]` because MLX graphics-processor tests never run
//! in parallel.

use astronomical_jangtq2_qwen35::codebook;
use astronomical_jangtq2_qwen35::hadamard32;
use astronomical_jangtq2_qwen35::mlx_decode::{
    decode_expert_weight_rows, rotate_activations_hadamard32,
};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use super::support::contract_runtime;

fn reference_golden_u32_values(key: &str) -> Vec<u32> {
    serde_json::from_str::<serde_json::Value>(include_str!(
        "../../src/golden_reference_vectors.json"
    ))
    .expect("the checked-in jangtq2 golden vectors should be valid JSON")[key]
        .as_array()
        .expect("the requested golden value should be a list")
        .iter()
        .map(|golden_value| {
            u32::try_from(golden_value.as_u64().expect("golden values are unsigned"))
                .expect("golden packed words fit uint32")
        })
        .collect()
}

#[test]
#[serial]
fn should_decode_a_multi_expert_page_identically_to_per_expert_pages() {
    let runtime = contract_runtime();
    // Forty experts on a 3-bit, 64-wide page: every row of the page is
    // decoded by its own threads, and a per-expert reference decode pins
    // bit-exact parity between a large grid and a one-expert grid, so
    // row indexing inside the kernel cannot drift with the page size.
    let expert_count = 40_usize;
    let output_rows = 2_usize;
    let packed_words_per_row = 6_usize;
    let decoded_width = packed_words_per_row * u32::BITS as usize / 3;
    let packed_words_host: Vec<u32> = (0..expert_count * output_rows * packed_words_per_row)
        .map(|word_index| {
            (word_index as u32)
                .wrapping_mul(2_654_435_761)
                .rotate_left(7)
        })
        .collect();
    let row_scales_host: Vec<f32> = (0..expert_count * output_rows)
        .map(|scale_index| ((scale_index % 7) as f32 + 1.0) / 4.0)
        .collect();
    let packed_page = MlxArray::from_u32(
        &packed_words_host,
        &[
            expert_count as i32,
            output_rows as i32,
            packed_words_per_row as i32,
        ],
    )
    .expect("the synthetic multi-block page should form a valid tensor");
    let page_scales = runtime
        .array_from_f32(&row_scales_host, &[expert_count as i32, output_rows as i32])
        .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
        .expect("the synthetic multi-block row scales should upload");

    let decoded_page = decode_expert_weight_rows(&runtime, &packed_page, &page_scales, 3)
        .expect("the multi-block page should decode");
    assert_eq!(
        decoded_page.shape(),
        [
            expert_count as i32,
            output_rows as i32,
            decoded_width as i32
        ]
    );
    let decoded_page_values = runtime
        .astype(&decoded_page, MlxDtype::Float32)
        .expect("the decoded page should convert for comparison")
        .to_vec_f32()
        .expect("the decoded page values should be readable");

    for expert_index in 0..expert_count {
        let expert_packed_words =
            &packed_words_host[expert_index * output_rows * packed_words_per_row
                ..(expert_index + 1) * output_rows * packed_words_per_row];
        let expert_packed_page = MlxArray::from_u32(
            expert_packed_words,
            &[1, output_rows as i32, packed_words_per_row as i32],
        )
        .expect("one expert's packed rows should form a valid tensor");
        let expert_row_scales =
            &row_scales_host[expert_index * output_rows..(expert_index + 1) * output_rows];
        let expert_scales = runtime
            .array_from_f32(expert_row_scales, &[1, output_rows as i32])
            .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
            .expect("one expert's row scales should upload");
        let decoded_expert =
            decode_expert_weight_rows(&runtime, &expert_packed_page, &expert_scales, 3)
                .expect("the per-expert reference page should decode");
        let decoded_expert_values = runtime
            .astype(&decoded_expert, MlxDtype::Float32)
            .expect("the per-expert decode should convert for comparison")
            .to_vec_f32()
            .expect("the per-expert decoded values should be readable");
        let page_offset = expert_index * output_rows * decoded_width;
        for (decoded_position, reference_value) in decoded_expert_values.iter().enumerate() {
            assert_eq!(
                decoded_page_values[page_offset + decoded_position],
                *reference_value,
                "expert {expert_index} must decode identically inside the whole page"
            );
        }
    }
}

#[test]
#[serial]
fn should_decode_reference_packed_three_bit_codes_on_mlx() {
    let runtime = contract_runtime();
    let packed_words = reference_golden_u32_values("pack3_random_words");
    let reference_codes = reference_golden_u32_values("pack3_random_codes");
    let packed_code_words = MlxArray::from_u32(&packed_words, &[1, 1, 3])
        .expect("the golden packed words should form one expert output row");
    let row_scales = runtime
        .array_from_f32(&[1.0], &[1, 1])
        .and_then(|float32_scales| runtime.astype(&float32_scales, MlxDtype::Float16))
        .expect("one float16 codebook row scale should be representable");

    let decoded_weight = decode_expert_weight_rows(&runtime, &packed_code_words, &row_scales, 3)
        .expect("the reference 3-bit expert row should decode");
    let decoded_float32_weight = runtime
        .astype(&decoded_weight, MlxDtype::Float32)
        .expect("the decoded bfloat16 row should convert for comparison");
    let decoded_values = decoded_float32_weight
        .to_vec_f32()
        .expect("the decoded float32 comparison row should be readable");

    assert_eq!(decoded_weight.shape(), [1, 1, 32]);
    assert_eq!(decoded_weight.dtype(), MlxDtype::BFloat16);
    for (code_index, (decoded_value, reference_code)) in
        decoded_values.iter().zip(reference_codes).enumerate()
    {
        let reference_level = codebook::codebook_level_value(3, reference_code)
            .expect("reference codes are supported 3-bit values");
        assert!(
            (decoded_value - reference_level).abs() < 0.01,
            "MLX decoded code {code_index} as {decoded_value}, expected {reference_level}"
        );
    }
}

#[test]
#[serial]
fn should_apply_blockwise_hadamard32_to_mlx_activations() {
    let runtime = contract_runtime();
    let activation_values = (0..64)
        .map(|input_index| (input_index as f32 - 17.0) / 11.0)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[1, 64])
        .expect("the 64-wide activation test tensor should be valid");
    let rotated_activations = rotate_activations_hadamard32(&runtime, &activations)
        .expect("both 32-wide activation blocks should rotate");
    let rotated_values = rotated_activations
        .to_vec_f32()
        .expect("the rotated activation rows should be readable");
    let expected_values = hadamard32::reference_transform_vector(&activation_values)
        .expect("the activation width contains two full Hadamard blocks");

    assert_eq!(rotated_activations.shape(), [1, 64]);
    for (actual_value, expected_value) in rotated_values.iter().zip(expected_values) {
        assert!(
            (actual_value - expected_value).abs() < 1e-4,
            "MLX Hadamard output {actual_value} differs from reference {expected_value}"
        );
    }
}
