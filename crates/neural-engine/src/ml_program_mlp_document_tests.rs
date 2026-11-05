use super::{MlpFp16Weights, encode_ml_program_mlp};
use crate::projection_model_interface::ProjectionDocumentError;
use crate::projection_shape::ProjectionShape;

const HIDDEN_CHANNELS: usize = 8;
const INTERMEDIATE_CHANNELS: usize = 16;
const TOKEN_COUNT: usize = 2;
const CONTRACTION_PART_WIDTH: usize = 4;

fn contains_subsequence(encoded: &[u8], expected_bytes: &[u8]) -> bool {
    for window in encoded.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

fn mlp_shape() -> ProjectionShape {
    ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, TOKEN_COUNT).expect("valid shape")
}

fn gate_and_up_weights() -> Vec<u16> {
    vec![0; 2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS]
}

fn down_weights() -> Vec<u16> {
    vec![0; HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS]
}

#[test]
fn should_write_the_gated_chain_with_each_contraction_split_into_parts() {
    let package = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
        },
        CONTRACTION_PART_WIDTH,
    )
    .expect("the MLP encodes");

    for expected_name in [
        b"sigmoid".as_slice(),
        b"gate_silu",
        b"gated",
        b"gate_and_up_partial_1",
        b"down_partial_3",
        b"down_sum_2",
    ] {
        assert!(
            contains_subsequence(&package.model_document, expected_name),
            "missing {}",
            String::from_utf8_lossy(expected_name)
        );
    }
}

#[test]
fn should_store_every_weight_as_fp16_in_the_blob_file() {
    let package = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
        },
        CONTRACTION_PART_WIDTH,
    )
    .expect("the MLP encodes");

    let weight_bytes = 2 * (gate_and_up_weights().len() + down_weights().len());
    assert!(package.weight_blob_file.len() >= weight_bytes);
}

#[test]
fn should_write_one_convolution_per_stage_when_a_part_spans_the_whole_contraction() {
    let package = encode_ml_program_mlp(
        ProjectionShape::new(4, 4, TOKEN_COUNT).expect("valid shape"),
        4,
        MlpFp16Weights {
            gate_and_up: &[0; 2 * 4 * 4],
            down: &[0; 4 * 4],
        },
        4,
    )
    .expect("the MLP encodes");

    assert!(!contains_subsequence(
        &package.model_document,
        b"down_partial"
    ));
    assert!(contains_subsequence(&package.model_document, b"sigmoid"));
}

#[test]
fn should_refuse_a_part_width_that_does_not_divide_the_hidden_channels() {
    let encoding_result = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
        },
        3,
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::UnevenContractionPartWidth {
            input_channels: HIDDEN_CHANNELS,
            contraction_part_width: 3,
        })
    );
}

#[test]
fn should_refuse_a_part_width_that_divides_the_hidden_but_not_the_intermediate_channels() {
    let encoding_result = encode_ml_program_mlp(
        mlp_shape(),
        12,
        MlpFp16Weights {
            gate_and_up: &[0; 2 * 12 * HIDDEN_CHANNELS],
            down: &[0; HIDDEN_CHANNELS * 12],
        },
        HIDDEN_CHANNELS,
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::UnevenContractionPartWidth {
            input_channels: 12,
            contraction_part_width: HIDDEN_CHANNELS,
        })
    );
}

#[test]
fn should_refuse_a_part_width_of_zero() {
    let encoding_result = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up_weights(),
            down: &down_weights(),
        },
        0,
    );

    assert!(matches!(
        encoding_result,
        Err(ProjectionDocumentError::UnevenContractionPartWidth { .. })
    ));
}

#[test]
fn should_refuse_gate_and_up_weights_of_the_wrong_length() {
    let encoding_result = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &[0; 5],
            down: &down_weights(),
        },
        CONTRACTION_PART_WIDTH,
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::WeightCountMismatch {
            expected: 2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS,
            actual: 5,
        })
    );
}

#[test]
fn should_refuse_down_weights_of_the_wrong_length() {
    let encoding_result = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up_weights(),
            down: &[0; 3],
        },
        CONTRACTION_PART_WIDTH,
    );

    assert_eq!(
        encoding_result.err(),
        Some(ProjectionDocumentError::WeightCountMismatch {
            expected: HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS,
            actual: 3,
        })
    );
}
