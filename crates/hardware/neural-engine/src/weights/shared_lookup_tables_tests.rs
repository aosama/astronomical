use super::{LOOKUP_TABLE_ENTRY_COUNT, SharedLookupTables};
use crate::documents::model_interface::ModelDocumentError;
use crate::weights::half_precision::{f16_bits_to_f32, f32_to_f16_bits};

fn fp16_ramp(value_count: usize) -> Vec<u16> {
    let mut fp16_values = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        fp16_values.push(f32_to_f16_bits(value_index as f32 * 0.25 - 40.0));
    }
    fp16_values
}

#[test]
fn should_reconstruct_ternary_weights_exactly_when_a_table_has_room() {
    let negative = f32_to_f16_bits(-0.012);
    let zero = f32_to_f16_bits(0.0);
    let positive = f32_to_f16_bits(0.012);
    let fp16_weights = vec![
        negative, zero, positive, zero, positive, negative, zero, zero,
    ];

    let encoded =
        SharedLookupTables::from_fp16_weights(&fp16_weights, 4, 2, 2).expect("the shape is even");

    assert_eq!(encoded.reconstructed_fp16(), fp16_weights);
    assert_eq!(encoded.merged_weight_count, 0);
    assert_eq!(encoded.fp16_tables.len(), 2 * LOOKUP_TABLE_ENTRY_COUNT);
}

#[test]
fn should_give_each_row_group_its_own_table() {
    let first_group_value = f32_to_f16_bits(1.5);
    let second_group_value = f32_to_f16_bits(-2.5);
    let fp16_weights = vec![
        first_group_value,
        first_group_value,
        second_group_value,
        second_group_value,
    ];

    let encoded =
        SharedLookupTables::from_fp16_weights(&fp16_weights, 2, 2, 1).expect("the shape is even");

    assert_eq!(encoded.fp16_tables[0], first_group_value);
    assert_eq!(
        encoded.fp16_tables[LOOKUP_TABLE_ENTRY_COUNT],
        second_group_value
    );
    assert_eq!(encoded.codes, vec![0, 0, 0, 0]);
}

#[test]
fn should_order_codes_by_value_across_the_sign() {
    let fp16_weights = vec![
        f32_to_f16_bits(2.0),
        f32_to_f16_bits(-1.0),
        f32_to_f16_bits(-3.0),
        f32_to_f16_bits(0.5),
    ];

    let encoded =
        SharedLookupTables::from_fp16_weights(&fp16_weights, 1, 4, 1).expect("the shape is even");

    assert_eq!(encoded.codes, vec![3, 1, 0, 2]);
}

#[test]
fn should_keep_exactly_a_full_table_without_merging() {
    let fp16_weights = fp16_ramp(LOOKUP_TABLE_ENTRY_COUNT);

    let encoded =
        SharedLookupTables::from_fp16_weights(&fp16_weights, 1, LOOKUP_TABLE_ENTRY_COUNT, 1)
            .expect("the shape is even");

    assert_eq!(encoded.reconstructed_fp16(), fp16_weights);
    assert_eq!(encoded.merged_weight_count, 0);
}

#[test]
fn should_merge_the_closest_values_when_a_table_overflows() {
    let mut fp16_weights = fp16_ramp(LOOKUP_TABLE_ENTRY_COUNT);
    // Closer to the ramp's first value than any two ramp values are together.
    let crowding_value = f32_to_f16_bits(-40.0 + 0.03125);
    fp16_weights.push(crowding_value);

    let encoded =
        SharedLookupTables::from_fp16_weights(&fp16_weights, 1, LOOKUP_TABLE_ENTRY_COUNT + 1, 1)
            .expect("the shape is even");

    assert_eq!(encoded.merged_weight_count, 2);
    let reconstructed = encoded.reconstructed_fp16();
    assert_eq!(reconstructed[0], reconstructed[LOOKUP_TABLE_ENTRY_COUNT]);
    // fp16 spacing near 40 is 0.03125, so the midpoint rounds to either value.
    assert!((f16_bits_to_f32(reconstructed[0]) - (-40.0 + 0.015_625)).abs() < 0.02);
    assert_eq!(reconstructed[1], fp16_weights[1]);
}

#[test]
fn should_handle_a_table_of_one_repeated_value() {
    let repeated_value = f32_to_f16_bits(0.25);

    let encoded = SharedLookupTables::from_fp16_weights(&[repeated_value; 6], 3, 2, 3)
        .expect("the shape is even");

    assert_eq!(encoded.codes, vec![0; 6]);
    assert_eq!(encoded.reconstructed_fp16(), vec![repeated_value; 6]);
}

#[test]
fn should_refuse_rows_per_table_that_do_not_divide_the_rows() {
    let encoding_result = SharedLookupTables::from_fp16_weights(&[0; 10], 5, 2, 2);

    assert_eq!(
        encoding_result.err(),
        Some(ModelDocumentError::UnevenLookupTableRows {
            output_channels: 5,
            rows_per_table: 2,
        })
    );
}

#[test]
fn should_refuse_zero_rows_per_table() {
    let encoding_result = SharedLookupTables::from_fp16_weights(&[0; 4], 2, 2, 0);

    assert!(matches!(
        encoding_result,
        Err(ModelDocumentError::UnevenLookupTableRows { .. })
    ));
}

#[test]
fn should_refuse_a_weight_count_that_does_not_match_the_shape() {
    let encoding_result = SharedLookupTables::from_fp16_weights(&[0; 3], 2, 2, 1);

    assert_eq!(
        encoding_result.err(),
        Some(ModelDocumentError::WeightCountMismatch {
            expected: 4,
            actual: 3,
        })
    );
}

#[test]
fn should_refuse_an_empty_part() {
    let encoding_result = SharedLookupTables::from_fp16_weights(&[], 2, 0, 1);

    assert!(matches!(
        encoding_result,
        Err(ModelDocumentError::WeightCountMismatch { .. })
    ));
}
