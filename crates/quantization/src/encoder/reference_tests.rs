//! Unit tests for the pure reference encoder. Numeric parity with the Python
//! tool on a fixed tensor lives in `tests/integration/encoder_parity.rs`.

use super::*;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("test bit widths are valid")
}

#[test]
#[ignore = "pending implementation of stock_min_max_parameters"]
fn should_derive_stock_parameters_from_the_group_extremes() {
    let group = [0.0_f32, 1.0, 2.0, 3.0];

    let parameters = stock_min_max_parameters(&group, bits(2));

    assert_eq!(parameters.scale, 1.0);
    assert_eq!(parameters.bias, 0.0);
    assert_eq!(parameters.squared_error, 0.0);
}

#[test]
#[ignore = "pending implementation of search_group_parameters"]
fn should_never_do_worse_than_stock_min_max() {
    // An outlier-heavy group where stock min/max collapses the bulk onto
    // one code; the search must achieve strictly lower squared error.
    let group = [0.01_f32, -0.02, 0.015, -0.01, 0.005, 0.0, -0.005, 0.9];

    let stock = stock_min_max_parameters(&group, bits(2));
    let searched = search_group_parameters(&group, bits(2));

    assert!(searched.squared_error <= stock.squared_error);
}

#[test]
#[ignore = "pending implementation of search_group_parameters"]
fn should_keep_a_true_zero_representable_when_the_edge_lands_on_code_zero() {
    // When round(edge / raw) is 0 the bias is set to exactly 0.0.
    let group = [0.0_f32, 0.0, 0.0, 1e-9];

    let searched = search_group_parameters(&group, bits(2));

    assert_eq!(searched.bias, 0.0);
}

#[test]
#[ignore = "pending implementation of pack_affine_codes"]
fn should_pack_four_bit_codes_low_to_high_eight_per_word() {
    // Element 0 occupies the four least significant bits, element 1 bits 4-7.
    let codes = [1_u32, 2, 3, 4, 5, 6, 7, 8];

    let packed = pack_affine_codes(&codes, 8, bits(4));

    assert_eq!(packed, vec![0x8765_4321]);
}

#[test]
#[ignore = "pending implementation of pack_affine_codes and unpack_affine_codes"]
fn should_round_trip_three_bit_codes_that_straddle_word_boundaries() {
    // 32 three-bit codes span 96 bits = 3 words; code 10 straddles words 0 and 1.
    let codes: Vec<u32> = (0..32).map(|index| index % 8).collect();

    let packed = pack_affine_codes(&codes, 32, bits(3));
    let unpacked = unpack_affine_codes(&packed, 1, 32, bits(3));

    assert_eq!(packed.len(), 3);
    assert_eq!(unpacked, codes);
}

#[test]
#[ignore = "pending implementation of encode_slab"]
fn should_reject_a_slab_whose_columns_do_not_divide_the_group_size() {
    let weights = vec![0.0_f32; 2 * 48];

    let encode_outcome = encode_slab(
        &weights,
        2,
        48,
        bits(4),
        GroupSize::new(64).expect("valid"),
        true,
    );

    assert!(matches!(
        encode_outcome,
        Err(QuantizationError::TensorNotGroupQuantizable { .. })
    ));
}

#[test]
#[ignore = "pending implementation of encode_slab and dequantize_slab"]
fn should_round_trip_within_one_scale_step_per_element() {
    let weights: Vec<f32> = (0..128).map(|index| (index as f32 - 64.0) / 64.0).collect();

    let slab = encode_slab(
        &weights,
        1,
        128,
        bits(4),
        GroupSize::new(64).expect("valid"),
        false,
    )
    .expect("a 1x128 slab is group-quantizable at 64");
    let reconstructed = dequantize_slab(&slab);

    for (original, reconstructed_value) in weights.iter().zip(&reconstructed) {
        assert!((original - reconstructed_value).abs() <= 2.0 / 15.0 + 1e-6);
    }
}

#[test]
#[ignore = "pending implementation of exact_zero_fraction"]
fn should_measure_the_exact_zero_fraction() {
    assert_eq!(exact_zero_fraction(&[0.0, 1.0, 0.0, 2.0]), 0.5);
    assert_eq!(exact_zero_fraction(&[]), 0.0);
}
