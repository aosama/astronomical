use super::{MlpFp16Weights, encode_ml_program_mlp, encode_ml_program_mlp_with_encoding};
use crate::documents::mlp_stages::StageWeightEncoding;
use crate::documents::model_interface::ModelDocumentError;
use crate::program_shape::ProgramShape;

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

fn mlp_shape() -> ProgramShape {
    ProgramShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, TOKEN_COUNT).expect("valid shape")
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
        ProgramShape::new(4, 4, TOKEN_COUNT).expect("valid shape"),
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
        Some(ModelDocumentError::UnevenContractionPartWidth {
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
        Some(ModelDocumentError::UnevenContractionPartWidth {
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
        Err(ModelDocumentError::UnevenContractionPartWidth { .. })
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
        Some(ModelDocumentError::WeightCountMismatch {
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
        Some(ModelDocumentError::WeightCountMismatch {
            expected: HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS,
            actual: 3,
        })
    );
}

/// Distinct small values, so a table has something to hold and the two encodings
/// differ in more than their headers.
fn varied_weights(weight_count: usize) -> Vec<u16> {
    let mut fp16_weights = Vec::with_capacity(weight_count);
    for weight_index in 0..weight_count {
        fp16_weights.push(0x3c00 + (weight_index % 7) as u16);
    }
    fp16_weights
}

fn encoded_with(weight_encoding: StageWeightEncoding) -> Result<Vec<u8>, ModelDocumentError> {
    let gate_and_up = varied_weights(2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS);
    let down = varied_weights(HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS);
    let package = encode_ml_program_mlp_with_encoding(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &gate_and_up,
            down: &down,
        },
        CONTRACTION_PART_WIDTH,
        weight_encoding,
    )?;
    Ok(package.weight_blob_file)
}

#[test]
fn should_write_the_same_program_as_the_fp16_entry_point_when_asked_for_fp16() {
    let gate_and_up = varied_weights(2 * INTERMEDIATE_CHANNELS * HIDDEN_CHANNELS);
    let down = varied_weights(HIDDEN_CHANNELS * INTERMEDIATE_CHANNELS);
    let weights = MlpFp16Weights {
        gate_and_up: &gate_and_up,
        down: &down,
    };

    let fp16_package = encode_ml_program_mlp(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        weights,
        CONTRACTION_PART_WIDTH,
    )
    .expect("the MLP encodes");
    let chosen_package = encode_ml_program_mlp_with_encoding(
        mlp_shape(),
        INTERMEDIATE_CHANNELS,
        weights,
        CONTRACTION_PART_WIDTH,
        StageWeightEncoding::Fp16,
    )
    .expect("the MLP encodes");

    assert_eq!(chosen_package.model_document, fp16_package.model_document);
    assert_eq!(
        chosen_package.weight_blob_file,
        fp16_package.weight_blob_file
    );
}

/// A 256-entry table costs 512 bytes, more than the tiny shape's whole fp16 part, so
/// the byte saving only shows on a part with far more weights than one table holds.
#[test]
fn should_store_about_half_the_weight_bytes_as_lookup_tables() {
    const WIDE_HIDDEN_CHANNELS: usize = 256;
    const WIDE_INTERMEDIATE_CHANNELS: usize = 512;
    let gate_and_up = varied_weights(2 * WIDE_INTERMEDIATE_CHANNELS * WIDE_HIDDEN_CHANNELS);
    let down = varied_weights(WIDE_HIDDEN_CHANNELS * WIDE_INTERMEDIATE_CHANNELS);
    let wide_shape = ProgramShape::new(WIDE_HIDDEN_CHANNELS, WIDE_HIDDEN_CHANNELS, TOKEN_COUNT)
        .expect("valid shape");
    let mut weight_file_byte_counts = Vec::new();
    for weight_encoding in [
        StageWeightEncoding::Fp16,
        StageWeightEncoding::SharedLookupTables {
            rows_per_table: 256,
        },
    ] {
        let package = encode_ml_program_mlp_with_encoding(
            wide_shape,
            WIDE_INTERMEDIATE_CHANNELS,
            MlpFp16Weights {
                gate_and_up: &gate_and_up,
                down: &down,
            },
            WIDE_HIDDEN_CHANNELS,
            weight_encoding,
        )
        .expect("the MLP encodes");
        weight_file_byte_counts.push(package.weight_blob_file.len());
    }

    let (fp16_bytes, lookup_table_bytes) = (weight_file_byte_counts[0], weight_file_byte_counts[1]);
    assert!(
        lookup_table_bytes * 100 < fp16_bytes * 55 && lookup_table_bytes * 100 > fp16_bytes * 45,
        "{lookup_table_bytes} lookup-table bytes against {fp16_bytes} fp16 bytes"
    );
}

#[test]
fn should_accept_one_table_per_row() {
    encoded_with(StageWeightEncoding::SharedLookupTables { rows_per_table: 1 })
        .expect("one table per row encodes");
}

#[test]
fn should_refuse_tables_that_do_not_divide_the_output_rows() {
    let encoding_error =
        encoded_with(StageWeightEncoding::SharedLookupTables { rows_per_table: 3 })
            .expect_err("three rows per table cannot tile eight or sixteen output rows");

    assert!(
        matches!(
            encoding_error,
            ModelDocumentError::UnevenLookupTableRows { .. }
        ),
        "{encoding_error:?}"
    );
}

#[test]
fn should_refuse_tables_of_no_rows() {
    encoded_with(StageWeightEncoding::SharedLookupTables { rows_per_table: 0 })
        .expect_err("a table must span at least one row");
}
