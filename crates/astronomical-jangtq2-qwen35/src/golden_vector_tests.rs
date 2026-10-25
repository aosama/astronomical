//! Golden-vector tests against the reference jangtq2 implementation.
//!
//! The fixture (`src/golden_reference_vectors.json`) was produced by
//! running the reference `format.py` under its own MLX environment:
//! codebook tables, packed words, a Hadamard-32 transform, and two complete
//! encode-then-dequant matrices. Regenerating the fixture requires that
//! repository; it is checked in so these tests stay offline and stable.
//!
//! All tests in this module are pure arithmetic — no GPU, no MLX — so they
//! are safe to run in parallel.

use serde_json::Value;

use crate::codebook;
use crate::hadamard32;

fn golden_vectors() -> Value {
    serde_json::from_str(include_str!("golden_reference_vectors.json"))
        .expect("the checked-in golden fixture is valid JSON")
}

fn golden_f32_list(key: &str) -> Vec<f32> {
    golden_vectors()[key]
        .as_array()
        .unwrap_or_else(|| panic!("golden key `{key}` holds an array"))
        .iter()
        .map(|list_value| list_value.as_f64().unwrap() as f32)
        .collect()
}

fn golden_f32_matrix(key: &str) -> Vec<Vec<f32>> {
    golden_vectors()[key]
        .as_array()
        .unwrap_or_else(|| panic!("golden key `{key}` holds a matrix"))
        .iter()
        .map(|row_value| {
            row_value
                .as_array()
                .expect("golden matrix rows are arrays")
                .iter()
                .map(|cell_value| cell_value.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn golden_u32_list(key: &str) -> Vec<u32> {
    golden_vectors()[key]
        .as_array()
        .unwrap_or_else(|| panic!("golden key `{key}` holds an array"))
        .iter()
        .map(|word_value| word_value.as_u64().unwrap() as u32)
        .collect()
}

fn golden_u8_matrix(key: &str) -> Vec<Vec<u8>> {
    golden_vectors()[key]
        .as_array()
        .unwrap_or_else(|| panic!("golden key `{key}` holds a matrix"))
        .iter()
        .map(|row_value| {
            row_value
                .as_array()
                .expect("golden code rows are arrays")
                .iter()
                .map(|code_value| code_value.as_u64().unwrap() as u8)
                .collect()
        })
        .collect()
}

#[test]
fn should_match_reference_codebook_levels_for_every_supported_bit_width() {
    for bit_width in codebook::SUPPORTED_CODEBOOK_BIT_WIDTHS {
        let reference_levels = golden_f32_list(&format!("codebook_bits_{bit_width}"));
        let implemented_levels = codebook::codebook_levels(*bit_width)
            .unwrap_or_else(|| panic!("bit width {bit_width} must be supported"));
        assert_eq!(implemented_levels, reference_levels);
        // The arithmetic evaluation must agree with the table bit-for-bit:
        // both compute in f64 and narrow once.
        for (code, reference_level) in reference_levels.iter().enumerate() {
            assert_eq!(
                codebook::codebook_level_value(*bit_width, code as u32),
                Some(*reference_level),
                "arithmetic level must equal the reference table at bits={bit_width} code={code}"
            );
        }
    }
}

#[test]
fn should_match_reference_cubic_parameters_for_every_supported_bit_width() {
    // A direct pin of the "do not re-fit" constants: the level tables above
    // would also catch a drift, but only after computing it away; this
    // assertion fails on the constant itself. The comparison tolerates one
    // unit in the last place because serde_json without its float_roundtrip
    // feature may parse a decimal literal a single ulp away from the value
    // Python wrote — an error far below the f32 level tables' resolution,
    // which is where the constants actually matter.
    let golden_document = golden_vectors();
    for bit_width in codebook::SUPPORTED_CODEBOOK_BIT_WIDTHS {
        let reference_pair = golden_document[&format!("cubic_params_bits_{bit_width}")]
            .as_array()
            .unwrap_or_else(|| panic!("golden cubic params for {bit_width} hold an array"));
        let reference_alpha = reference_pair[0].as_f64().unwrap();
        let reference_beta = reference_pair[1].as_f64().unwrap();
        let (alpha, beta) = codebook::cubic_parameters(*bit_width)
            .unwrap_or_else(|| panic!("bit width {bit_width} must be supported"));
        assert!(
            (alpha - reference_alpha).abs() <= f64::EPSILON * reference_alpha.abs(),
            "alpha {alpha} drifted from the reference {reference_alpha}"
        );
        assert!(
            (beta - reference_beta).abs() <= f64::EPSILON * reference_beta.abs(),
            "beta {beta} drifted from the reference {reference_beta}"
        );
    }
}

#[test]
fn should_reject_bit_widths_outside_the_codebook_family() {
    for unsupported_bit_width in [0, 1, 5, 6, 8, 16] {
        assert!(!codebook::is_supported_bit_width(unsupported_bit_width));
        assert!(codebook::cubic_parameters(unsupported_bit_width).is_none());
        assert!(codebook::codebook_levels(unsupported_bit_width).is_none());
    }
}

#[test]
fn should_unpack_reference_packed_words_to_reference_codes() {
    let reference_words = golden_u32_list("pack3_random_words");
    let reference_codes = golden_u32_list("pack3_random_codes");
    let unpacked_codes = codebook::unpack_codes_from_packed_row(&reference_words, 3)
        .expect("a reference 3-bit row must unpack");
    assert_eq!(
        unpacked_codes,
        reference_codes
            .iter()
            .map(|&code| code as u8)
            .collect::<Vec<_>>()
    );
    // The reference's own round-trip output must match ours as well.
    let reference_roundtrip = golden_u32_list("unpack3_roundtrip");
    assert_eq!(
        unpacked_codes,
        reference_roundtrip
            .iter()
            .map(|&code| code as u8)
            .collect::<Vec<_>>()
    );
}

#[test]
fn should_unpack_reference_two_bit_pattern_words() {
    // The fixture packed the repeating pattern 0,1,2,3 over one 32-code row
    // (2 bits): codes cycle, so unpacking must reproduce the cycle exactly.
    let reference_words = golden_u32_list("pack2_codes_0to3_repeat");
    let unpacked_codes = codebook::unpack_codes_from_packed_row(&reference_words, 2)
        .expect("a reference 2-bit row must unpack");
    let expected_pattern: Vec<u8> = (0..32).map(|code_index| (code_index % 4) as u8).collect();
    assert_eq!(unpacked_codes, expected_pattern);
}

#[test]
fn should_reject_malformed_packed_rows() {
    assert!(codebook::unpack_codes_from_packed_row(&[], 2).is_none());
    // Five words at 2 bits cannot be a whole number of 32-code groups.
    assert!(codebook::unpack_codes_from_packed_row(&[0; 5], 2).is_none());
    assert!(codebook::unpack_codes_from_packed_row(&[0; 8], 5).is_none());
}

#[test]
fn should_derive_decoded_weight_page_shapes_from_packed_codebook_rows() {
    assert_eq!(
        codebook::decoded_weight_page_shape(&[256, 1024, 192], 2, 8),
        Some(vec![8, 1024, 3072])
    );
    assert_eq!(
        codebook::decoded_weight_page_shape(&[256, 3072, 96], 3, 16),
        Some(vec![16, 3072, 1024])
    );
    assert!(codebook::decoded_weight_page_shape(&[256, 1024, 192], 2, 257).is_none());
    assert!(codebook::decoded_weight_page_shape(&[256, 1024, 193], 2, 8).is_none());
    assert!(codebook::decoded_weight_page_shape(&[256, 1024, 192], 5, 8).is_none());
}

#[test]
fn should_match_reference_hadamard32_transform_on_the_golden_vector() {
    let golden_input = golden_f32_list("h32_input");
    let golden_output = golden_f32_list("h32_output");
    let transformed_input = hadamard32::reference_transform_vector(&golden_input)
        .expect("the golden input width is blockable");
    // Reordered f32 summation (direct dot product vs the reference's
    // butterfly) differs by a few ulps; the comparison tolerates that while
    // still pinning the normalization convention (1/sqrt(32)) tightly.
    for (transformed_value, reference_value) in transformed_input.iter().zip(&golden_output) {
        let allowed_error = 1e-5 * reference_value.abs().max(1.0);
        assert!(
            (transformed_value - reference_value).abs() <= allowed_error,
            "transform {transformed_value} must match reference {reference_value}"
        );
    }
}

#[test]
fn should_be_self_inverse_and_norm_preserving_on_the_golden_vector() {
    let golden_input = golden_f32_list("h32_input");
    let once_transformed = hadamard32::reference_transform_vector(&golden_input)
        .expect("the golden input width is blockable");
    let twice_transformed = hadamard32::reference_transform_vector(&once_transformed)
        .expect("the transformed width stays blockable");
    for (double_transformed_value, original_value) in twice_transformed.iter().zip(&golden_input) {
        assert!(
            (double_transformed_value - original_value).abs() <= 1e-5,
            "h32 must be its own inverse"
        );
    }
    let input_squared_norm: f32 = golden_input.iter().map(|value| value * value).sum();
    let transformed_squared_norm: f32 = once_transformed.iter().map(|value| value * value).sum();
    assert!(
        (input_squared_norm - transformed_squared_norm).abs() <= 1e-4 * input_squared_norm.max(1.0),
        "h32 must preserve the norm (orthonormal blocks)"
    );
}

#[test]
fn should_decode_reference_two_bit_codes_with_reference_scales() {
    should_decode_reference_codes_with_reference_scales_for_bit_width(2);
}

#[test]
fn should_decode_reference_three_bit_codes_with_reference_scales() {
    should_decode_reference_codes_with_reference_scales_for_bit_width(3);
}

fn should_decode_reference_codes_with_reference_scales_for_bit_width(bit_width: u32) {
    // End-to-end decode check: codes + per-row scales through this crate's
    // arithmetic codebook must reproduce the reference dequant matrix
    // bit-for-bit (both compute level(code) * scale in f32).
    let reference_codes = golden_u8_matrix(&format!("encode{bit_width}_codes"));
    let reference_scales = golden_f32_list(&format!("encode{bit_width}_scales"));
    let reference_dequant = golden_f32_matrix(&format!("encode{bit_width}_dequant"));
    for (row_index, (code_row, reference_row)) in
        reference_codes.iter().zip(&reference_dequant).enumerate()
    {
        let row_scale = reference_scales[row_index];
        for (code, reference_value) in code_row.iter().zip(reference_row) {
            let decoded_value = codebook::codebook_level_value(bit_width, u32::from(*code))
                .expect("reference codes are always in range")
                * row_scale;
            assert_eq!(
                decoded_value, *reference_value,
                "decode must match the reference at row {row_index}"
            );
        }
    }
}
