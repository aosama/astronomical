use super::{
    RotatedMlpLayout, RotatedMlpWeights, encode_ml_program_rotated_mlp, normalized_hadamard_fp16,
};
use crate::half_precision::{f16_bits_to_f32, f32_to_f16_bits};
use crate::ml_program_mlp_document::StageWeightEncoding;
use crate::projection_model_interface::ProjectionDocumentError;
use crate::projection_shape::ProjectionShape;
use crate::shared_lookup_tables::LOOKUP_TABLE_ENTRY_COUNT;

const HIDDEN_CHANNELS: usize = 8;
const INTERMEDIATE_CHANNELS: usize = 16;
const PART_WIDTH: usize = 4;
const ROTATION_BLOCK_SIZE: usize = 4;
const TOKEN_COUNT: usize = 2;
const ROWS_PER_TABLE: usize = 4;

fn contains_subsequence(encoded: &[u8], expected_bytes: &[u8]) -> bool {
    for window in encoded.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

fn layout() -> RotatedMlpLayout {
    RotatedMlpLayout {
        shape: ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, TOKEN_COUNT)
            .expect("valid shape"),
        intermediate_channels: INTERMEDIATE_CHANNELS,
        contraction_part_width: PART_WIDTH,
        rotation_block_size: ROTATION_BLOCK_SIZE,
        weight_encoding: StageWeightEncoding::SharedLookupTables {
            rows_per_table: ROWS_PER_TABLE,
        },
    }
}

fn gate_and_up_weights() -> Vec<u16> {
    vec![f32_to_f16_bits(0.01); 2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS]
}

fn down_weights() -> Vec<u16> {
    vec![f32_to_f16_bits(-0.01); HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS]
}

fn signs() -> Vec<u16> {
    vec![f32_to_f16_bits(1.0); INTERMEDIATE_CHANNELS]
}

#[test]
fn should_rotate_the_gated_activations_before_the_down_projection() {
    let package = encode_ml_program_rotated_mlp(
        layout(),
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    )
    .expect("the MLP encodes");

    for expected_name in [
        b"gate_silu".as_slice(),
        b"signed_gated",
        b"rotated_block_3",
        b"concat",
        b"constexpr_lut_to_dense",
    ] {
        assert!(
            contains_subsequence(&package.model_document, expected_name),
            "missing {}",
            String::from_utf8_lossy(expected_name)
        );
    }
}

#[test]
fn should_write_plain_fp16_weights_when_asked_to() {
    let mut fp16_layout = layout();
    fp16_layout.weight_encoding = StageWeightEncoding::Fp16;

    let package = encode_ml_program_rotated_mlp(
        fp16_layout,
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    )
    .expect("the MLP encodes");

    assert!(!contains_subsequence(
        &package.model_document,
        b"constexpr_lut_to_dense"
    ));
}

#[test]
fn should_store_every_part_codes_and_tables_when_using_shared_tables() {
    let package = encode_ml_program_rotated_mlp(
        layout(),
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    )
    .expect("the MLP encodes");

    let code_bytes = gate_and_up_weights().len() + down_weights().len();
    let gate_and_up_table_count =
        2 * INTERMEDIATE_CHANNELS / ROWS_PER_TABLE * (HIDDEN_CHANNELS / PART_WIDTH);
    let down_table_count = HIDDEN_CHANNELS / ROWS_PER_TABLE * (INTERMEDIATE_CHANNELS / PART_WIDTH);
    let table_bytes =
        (gate_and_up_table_count + down_table_count) * LOOKUP_TABLE_ENTRY_COUNT * size_of::<u16>();
    assert!(package.weight_blob_file.len() >= code_bytes + table_bytes);
}

#[test]
fn should_build_a_two_by_two_hadamard_with_the_sylvester_sign_pattern() {
    let matrix = normalized_hadamard_fp16(2);
    let positive = 1.0 / 2.0_f32.sqrt();

    let expected_signs = [1.0, 1.0, 1.0, -1.0];
    for (entry, expected_sign) in matrix.iter().zip(expected_signs) {
        assert!((f16_bits_to_f32(*entry) - expected_sign * positive).abs() < 1e-3);
    }
}

#[test]
fn should_build_an_orthonormal_hadamard_at_the_real_block_size() {
    let block_size = 32;
    let matrix = normalized_hadamard_fp16(block_size);

    for first_row in 0..block_size {
        for second_row in 0..block_size {
            let mut dot_product = 0.0_f32;
            for column in 0..block_size {
                dot_product += f16_bits_to_f32(matrix[first_row * block_size + column])
                    * f16_bits_to_f32(matrix[second_row * block_size + column]);
            }
            let expected = if first_row == second_row { 1.0 } else { 0.0 };
            assert!((dot_product - expected).abs() < 1e-2);
        }
    }
}

#[test]
fn should_refuse_a_rotation_block_that_does_not_divide_the_intermediate_channels() {
    let mut uneven_layout = layout();
    uneven_layout.rotation_block_size = 5;

    let encoding_result = encode_ml_program_rotated_mlp(
        uneven_layout,
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::UnevenRotationBlocks {
            channel_count: INTERMEDIATE_CHANNELS,
            block_size: 5,
        })
    );
}

#[test]
fn should_refuse_a_rotation_block_of_zero() {
    let mut zero_layout = layout();
    zero_layout.rotation_block_size = 0;

    let encoding_result = encode_ml_program_rotated_mlp(
        zero_layout,
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    );

    assert!(matches!(
        encoding_result,
        Err(ProjectionDocumentError::UnevenRotationBlocks { .. })
    ));
}

#[test]
fn should_refuse_a_rotation_block_that_is_not_a_power_of_two() {
    let mut non_hadamard_layout = layout();
    non_hadamard_layout.intermediate_channels = 12;
    non_hadamard_layout.rotation_block_size = 6;

    let encoding_result = encode_ml_program_rotated_mlp(
        non_hadamard_layout,
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &[f32_to_f16_bits(1.0); 12],
        },
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::UnevenRotationBlocks {
            channel_count: 12,
            block_size: 6,
        })
    );
}

#[test]
fn should_refuse_signs_of_the_wrong_length() {
    let encoding_result = encode_ml_program_rotated_mlp(
        layout(),
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &[f32_to_f16_bits(1.0); 3],
        },
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::WeightCountMismatch {
            expected: INTERMEDIATE_CHANNELS,
            actual: 3,
        })
    );
}

#[test]
fn should_refuse_table_rows_that_do_not_divide_the_output_rows() {
    let mut uneven_layout = layout();
    uneven_layout.weight_encoding = StageWeightEncoding::SharedLookupTables { rows_per_table: 5 };

    let encoding_result = encode_ml_program_rotated_mlp(
        uneven_layout,
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
            intermediate_signs: &signs(),
        },
    );

    assert!(matches!(
        encoding_result,
        Err(ProjectionDocumentError::UnevenLookupTableRows { .. })
    ));
}
