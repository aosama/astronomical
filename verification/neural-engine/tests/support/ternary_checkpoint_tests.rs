use astronomical_neural_engine::weights::half_precision::f32_to_f16_bits;

use super::{
    BLOCK_INPUT_COLUMNS, TernaryProjection, little_endian_f32s, little_endian_u16s,
    little_endian_u32s,
};

#[test]
fn should_decode_little_endian_words_and_halves() {
    assert_eq!(
        little_endian_u32s(&[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]),
        [1, 0x8000_0000]
    );
    assert_eq!(little_endian_u16s(&[0x00, 0x3c, 0x01, 0x00]), [0x3c00, 1]);
    assert_eq!(little_endian_f32s(&1.5_f32.to_le_bytes()), [1.5]);
}

#[test]
fn should_decode_nothing_from_no_bytes() {
    assert!(little_endian_u32s(&[]).is_empty());
    assert!(little_endian_u16s(&[]).is_empty());
    assert!(little_endian_f32s(&[]).is_empty());
}

#[test]
fn should_ignore_a_trailing_partial_value() {
    assert_eq!(little_endian_u32s(&[1, 0, 0, 0, 9, 9]), [1]);
    assert_eq!(little_endian_u16s(&[1, 0, 9]), [1]);
}

#[test]
fn should_apply_each_block_scale_and_bias_to_its_own_codes() {
    let input_channels = 2 * BLOCK_INPUT_COLUMNS;
    let words_per_row = input_channels / 16;
    let mut packed_codes = vec![0_u32; words_per_row];
    // Code 3 in the row's first column and code 1 in the second block's first.
    packed_codes[0] = 0b11;
    packed_codes[words_per_row / 2] = 0b01;
    let projection = TernaryProjection {
        row_count: 1,
        input_channels,
        packed_codes,
        fp16_scales: vec![f32_to_f16_bits(1.0), f32_to_f16_bits(2.0)],
        fp16_biases: vec![f32_to_f16_bits(-1.0), f32_to_f16_bits(0.5)],
        signs: Vec::new(),
    };

    let fp16_weights = projection.fp16_weights();

    assert_eq!(projection.blocks_per_row(), 2);
    assert_eq!(projection.packed_words_per_row(), words_per_row);
    assert_eq!(fp16_weights.len(), input_channels);
    assert_eq!(fp16_weights[0], f32_to_f16_bits(2.0));
    assert_eq!(fp16_weights[1], f32_to_f16_bits(-1.0));
    assert_eq!(fp16_weights[BLOCK_INPUT_COLUMNS], f32_to_f16_bits(2.5));
    assert_eq!(fp16_weights[BLOCK_INPUT_COLUMNS + 1], f32_to_f16_bits(0.5));
}

#[test]
fn should_produce_no_weights_for_a_projection_without_rows() {
    let projection = TernaryProjection {
        row_count: 0,
        input_channels: BLOCK_INPUT_COLUMNS,
        packed_codes: Vec::new(),
        fp16_scales: Vec::new(),
        fp16_biases: Vec::new(),
        signs: Vec::new(),
    };

    assert!(projection.fp16_weights().is_empty());
}
