use super::{AffineQuantizationError, AffineQuantizedWeights, QuantizationLayout};
use crate::deterministic_values::{PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values};
use crate::weights::half_precision::{f16_bits_slice_to_f32, f16_bits_to_f32, f32_to_f16_bits};

const CHECKPOINT: QuantizationLayout = QuantizationLayout::CHECKPOINT_INT4;

fn row_groups(code_bits: u8, block_input_columns: usize) -> QuantizationLayout {
    QuantizationLayout {
        code_bits,
        block_output_rows: 1,
        block_input_columns,
    }
}

fn quantized_bell_shaped_weights(
    output_channels: usize,
    input_channels: usize,
    layout: QuantizationLayout,
) -> AffineQuantizedWeights {
    let weights = bell_shaped_values(
        0x0161,
        output_channels * input_channels,
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    );
    AffineQuantizedWeights::quantize(&weights, output_channels, input_channels, layout)
        .expect("the layout is valid")
}

fn evenly_spaced_weights(code_count: u16) -> Vec<f32> {
    // min −1 in steps of 0.125.
    let mut weights = Vec::new();
    for code in 0..code_count {
        weights.push(-1.0 + 0.125 * f32::from(code));
    }
    weights
}

#[test]
fn should_reproduce_a_group_that_is_already_on_the_code_grid() {
    // min 0, max 15 → scale 1, bias 0, so every integer weight is exact.
    let grid_weights: Vec<f32> = (0..16_u8).map(f32::from).collect();

    let quantized = AffineQuantizedWeights::quantize(&grid_weights, 1, 16, row_groups(4, 16))
        .expect("valid layout");

    assert_eq!(quantized.codes(), (0..16).collect::<Vec<u8>>().as_slice());
    assert_eq!(
        f16_bits_slice_to_f32(&quantized.dequantized_fp16()),
        grid_weights
    );
}

#[test]
fn should_keep_every_code_within_the_code_width() {
    let four_bit = quantized_bell_shaped_weights(8, 256, CHECKPOINT);
    let eight_bit = quantized_bell_shaped_weights(8, 256, row_groups(8, 64));

    assert_eq!(four_bit.codes().iter().max(), Some(&15));
    assert_eq!(eight_bit.codes().iter().max(), Some(&255));
    assert_eq!(four_bit.fp16_scales().len(), 8 * 256 / 64);
}

#[test]
fn should_stay_within_half_a_step_of_the_original_weights() {
    let original = bell_shaped_values(0x0161, 4 * 128, PROJECTION_WEIGHT_STANDARD_DEVIATION);
    let quantized =
        AffineQuantizedWeights::quantize(&original, 4, 128, CHECKPOINT).expect("valid layout");

    let dequantized = f16_bits_slice_to_f32(&quantized.dequantized_fp16());

    for (weight_index, (original_weight, dequantized_weight)) in
        original.iter().zip(&dequantized).enumerate()
    {
        let scale = f16_bits_to_f32(quantized.fp16_scales()[weight_index / 64]);
        // Half a step plus fp16 rounding of the scale, bias, and result.
        assert!((original_weight - dequantized_weight).abs() <= 0.55 * scale);
    }
}

#[test]
fn should_share_one_scale_across_a_block_of_several_rows() {
    let layout = QuantizationLayout {
        code_bits: 4,
        block_output_rows: 2,
        block_input_columns: 64,
    };

    let quantized = quantized_bell_shaped_weights(4, 128, layout);

    assert_eq!(quantized.block_grid(), [2, 2]);
    assert_eq!(quantized.fp16_scales().len(), 4);
    assert_eq!(quantized.block_lookup_tables_fp16().len(), 4 * 16);
}

#[test]
fn should_give_a_constant_group_a_finite_offset() {
    let quantized =
        AffineQuantizedWeights::quantize(&[0.25; 64], 1, 64, CHECKPOINT).expect("valid layout");

    let offset = f16_bits_to_f32(quantized.shift_scale_offsets_fp16()[0]);

    assert!(offset.is_finite());
    assert_eq!(quantized.codes(), &[0; 64]);
    assert_eq!(
        f16_bits_slice_to_f32(&quantized.dequantized_fp16()),
        vec![0.25; 64]
    );
}

#[test]
fn should_reconstruct_nearly_the_same_weights_through_shift_and_scale() {
    let quantized = quantized_bell_shaped_weights(16, 512, CHECKPOINT);

    let checkpoint_weights = f16_bits_slice_to_f32(&quantized.dequantized_fp16());
    let core_ml_weights = f16_bits_slice_to_f32(&quantized.shift_scale_dequantized_fp16());

    for (weight_index, (checkpoint_weight, core_ml_weight)) in
        checkpoint_weights.iter().zip(&core_ml_weights).enumerate()
    {
        let scale = f16_bits_to_f32(quantized.fp16_scales()[weight_index / 64]);
        // Zero-centred groups put the offset near 7.5, where an fp16 ulp is
        // 1/256; the fp16 rounding of the result adds at most one more ulp.
        let allowed_deviation = scale / 64.0 + checkpoint_weight.abs() / 1024.0;
        assert!((checkpoint_weight - core_ml_weight).abs() <= allowed_deviation);
    }
}

#[test]
fn should_reproduce_the_checkpoint_exactly_through_the_lookup_tables() {
    for layout in [CHECKPOINT, row_groups(8, 128)] {
        let quantized = quantized_bell_shaped_weights(4, 256, layout);

        let lookup_tables = quantized.block_lookup_tables_fp16();
        let mut looked_up_weights = Vec::with_capacity(quantized.codes().len());
        for (weight_index, code) in quantized.codes().iter().enumerate() {
            let block_index = weight_index / layout.block_input_columns;
            looked_up_weights
                .push(lookup_tables[block_index * layout.code_count() + usize::from(*code)]);
        }

        assert_eq!(looked_up_weights, quantized.dequantized_fp16());
    }
}

#[test]
fn should_fill_a_lookup_table_with_bias_plus_code_steps() {
    let weights = evenly_spaced_weights(16);

    let quantized =
        AffineQuantizedWeights::quantize(&weights, 1, 16, row_groups(4, 16)).expect("valid layout");

    assert_eq!(
        f16_bits_slice_to_f32(&quantized.block_lookup_tables_fp16()),
        weights
    );
}

#[test]
fn should_express_the_bias_as_a_negative_offset_in_steps() {
    // min −1, max 0.875 → scale 0.125, bias −1 → offset +8.
    let quantized =
        AffineQuantizedWeights::quantize(&evenly_spaced_weights(16), 1, 16, row_groups(4, 16))
            .expect("valid layout");

    assert_eq!(
        quantized.shift_scale_offsets_fp16(),
        vec![f32_to_f16_bits(8.0)]
    );
}

#[test]
fn should_report_the_code_range_of_each_width() {
    assert_eq!(row_groups(4, 64).maximum_code(), 15);
    assert_eq!(row_groups(4, 64).code_count(), 16);
    assert_eq!(row_groups(8, 64).maximum_code(), 255);
    assert_eq!(row_groups(8, 64).code_count(), 256);
}

#[test]
fn should_saturate_rather_than_overflow_for_unsupported_widths() {
    assert_eq!(row_groups(0, 64).maximum_code(), 0);
    assert_eq!(row_groups(9, 64).maximum_code(), 255);
    assert_eq!(row_groups(u8::MAX, 64).maximum_code(), 255);
}

#[test]
fn should_cut_a_part_spanning_row_blocks_on_its_column_boundaries() {
    let layout = QuantizationLayout {
        code_bits: 8,
        block_output_rows: 2,
        block_input_columns: 64,
    };
    let quantized = quantized_bell_shaped_weights(4, 128, layout);

    let first_half = quantized.contraction_part(0, 64);

    assert_eq!(first_half.block_grid(), [2, 1]);
    assert_eq!(
        first_half.fp16_scales(),
        &[quantized.fp16_scales()[0], quantized.fp16_scales()[2]]
    );
}

#[test]
fn should_cut_codes_and_scales_on_group_boundaries() {
    let quantized = quantized_bell_shaped_weights(2, 256, CHECKPOINT);

    let second_half = quantized.contraction_part(128, 128);

    assert_eq!(second_half.input_channels(), 128);
    assert_eq!(second_half.output_channels(), 2);
    assert_eq!(&second_half.codes()[..128], &quantized.codes()[128..256]);
    assert_eq!(&second_half.codes()[128..], &quantized.codes()[384..512]);
    assert_eq!(
        second_half.fp16_scales(),
        &[
            quantized.fp16_scales()[2],
            quantized.fp16_scales()[3],
            quantized.fp16_scales()[6],
            quantized.fp16_scales()[7],
        ]
    );
}

#[test]
#[should_panic(expected = "the part must cover whole quantization blocks inside the row")]
fn should_refuse_a_part_that_splits_a_quantization_block() {
    let quantized = quantized_bell_shaped_weights(1, 128, CHECKPOINT);

    let _ = quantized.contraction_part(32, 64);
}

#[test]
#[should_panic(expected = "the part must cover whole quantization blocks inside the row")]
fn should_refuse_a_part_that_runs_past_the_row() {
    let quantized = quantized_bell_shaped_weights(1, 128, CHECKPOINT);

    let _ = quantized.contraction_part(64, 128);
}

#[test]
fn should_reject_blocks_that_do_not_tile_the_matrix() {
    let uneven_columns = AffineQuantizedWeights::quantize(&[0.0; 96], 1, 96, CHECKPOINT);
    let uneven_rows = AffineQuantizedWeights::quantize(
        &[0.0; 192],
        3,
        64,
        QuantizationLayout {
            code_bits: 4,
            block_output_rows: 2,
            block_input_columns: 64,
        },
    );

    assert_eq!(
        uneven_columns,
        Err(AffineQuantizationError::UnevenBlocks {
            output_channels: 1,
            input_channels: 96,
            layout: CHECKPOINT
        })
    );
    assert!(uneven_rows.is_err());
    assert!(AffineQuantizedWeights::quantize(&[0.0; 64], 1, 64, row_groups(4, 0)).is_err());
}

#[test]
fn should_reject_a_code_width_other_than_four_or_eight_bits() {
    assert!(AffineQuantizedWeights::quantize(&[0.0; 64], 1, 64, row_groups(3, 64)).is_err());
    assert!(AffineQuantizedWeights::quantize(&[0.0; 64], 1, 64, row_groups(0, 64)).is_err());
}

#[test]
fn should_reject_a_weight_count_that_does_not_match_the_layout() {
    assert_eq!(
        AffineQuantizedWeights::quantize(&[0.0; 63], 1, 64, CHECKPOINT),
        Err(AffineQuantizationError::WeightCountMismatch {
            output_channels: 1,
            input_channels: 64,
            actual: 63
        })
    );
}
