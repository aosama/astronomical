use super::{MlProgramWeights, encode_ml_program_projection};
use crate::documents::model_interface::ModelDocumentError;
use crate::program_shape::ProgramShape;
use crate::weights::affine_quantized::{AffineQuantizedWeights, QuantizationLayout};
use astronomical_test_harness::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};

const SINGLE_CONVOLUTION: usize = 1;

fn contains_subsequence(encoded: &[u8], expected_bytes: &[u8]) -> bool {
    for window in encoded.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

fn quantized_weights(output_channels: usize, input_channels: usize) -> AffineQuantizedWeights {
    quantized_weights_in_layout(
        output_channels,
        input_channels,
        QuantizationLayout::CHECKPOINT_INT4,
    )
}

fn quantized_weights_in_layout(
    output_channels: usize,
    input_channels: usize,
    layout: QuantizationLayout,
) -> AffineQuantizedWeights {
    let weights = bell_shaped_values(
        0x0162,
        output_channels * input_channels,
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    );
    AffineQuantizedWeights::quantize(&weights, output_channels, input_channels, layout)
        .expect("valid layout")
}

#[test]
fn should_store_eight_bit_codes_with_a_256_entry_table_per_row_block() {
    let shape = ProgramShape::new(128, 4, 2).expect("valid shape");
    let weights = quantized_weights_in_layout(
        4,
        128,
        QuantizationLayout {
            code_bits: 8,
            block_output_rows: 2,
            block_input_columns: 128,
        },
    );

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::LookupTable(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    // Header, 512 one-byte indices, then 2 blocks × 256 fp16 entries.
    assert_eq!(
        contents.weight_blob_file.len(),
        64 + (64 + 512) + (64 + 1024)
    );
    // LUT shape [2, 1, 1, 1, 256, 1]: the 256 is varint 0x80 0x02.
    assert!(contains_subsequence(
        &contents.model_document,
        &[0x0a, 0x03, 0x08, 0x80, 0x02]
    ));
}

#[test]
fn should_start_with_the_block_wise_capable_specification_version() {
    let shape = ProgramShape::new(2, 3, 4).expect("valid shape");

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::Fp16(&[0x3c00; 6]),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    assert_eq!(&contents.model_document[..2], &[0x08, 0x09]);
    assert!(contains_subsequence(&contents.model_document, b"CoreML8"));
}

#[test]
fn should_bind_the_features_the_native_bridge_expects() {
    let shape = ProgramShape::new(2, 3, 4).expect("valid shape");

    let contents =
        encode_ml_program_projection(shape, MlProgramWeights::Fp16(&[0; 6]), SINGLE_CONVOLUTION)
            .expect("program encodes");

    assert!(contains_subsequence(
        &contents.model_document,
        b"activations"
    ));
    assert!(contains_subsequence(&contents.model_document, b"projected"));
    assert!(contains_subsequence(&contents.model_document, b"main"));
}

#[test]
fn should_keep_fp16_weights_in_the_blob_file_not_the_document() {
    let shape = ProgramShape::new(2, 1, 1).expect("valid shape");

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::Fp16(&[0x3c00, 0xc100]),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    assert!(contains_subsequence(
        &contents.weight_blob_file,
        &[0x00, 0x3c, 0x00, 0xc1]
    ));
    assert!(!contains_subsequence(
        &contents.model_document,
        &[0x00, 0x3c, 0x00, 0xc1]
    ));
    assert!(contains_subsequence(
        &contents.model_document,
        b"@model_path/weights/weight.bin"
    ));
}

#[test]
fn should_store_four_bit_codes_with_one_scale_and_offset_per_group() {
    let shape = ProgramShape::new(128, 8, 4).expect("valid shape");
    let weights = quantized_weights(8, 128);

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::BlockwiseShiftScale(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    assert!(contains_subsequence(
        &contents.model_document,
        b"constexpr_blockwise_shift_scale"
    ));
    // Header, then codes (512 B), scales (32 B), offsets (32 B), each behind
    // a 64-byte metadata record and padded to 64 bytes.
    assert_eq!(
        contents.weight_blob_file.len(),
        64 + (64 + 512) + (64 + 64) + (64 + 64)
    );
}

#[test]
fn should_store_four_bit_indices_with_one_sixteen_entry_table_per_group() {
    let shape = ProgramShape::new(128, 8, 4).expect("valid shape");
    let weights = quantized_weights(8, 128);

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::LookupTable(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    assert!(contains_subsequence(
        &contents.model_document,
        b"constexpr_lut_to_dense"
    ));
    // Header, indices (512 B), then 16 groups × 16 fp16 entries (512 B).
    assert_eq!(
        contents.weight_blob_file.len(),
        64 + (64 + 512) + (64 + 512)
    );
}

#[test]
fn should_make_a_four_bit_program_about_a_quarter_the_fp16_size() {
    let shape = ProgramShape::new(1024, 256, 4).expect("valid shape");
    let weights = quantized_weights(256, 1024);
    let fp16_weights = weights.dequantized_fp16();

    let fp16_contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::Fp16(&fp16_weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");
    let int4_contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::BlockwiseShiftScale(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    // 4 bits per weight plus 32 bits of scale and offset per 64 weights.
    let size_ratio =
        int4_contents.weight_blob_file.len() as f64 / fp16_contents.weight_blob_file.len() as f64;
    assert!((0.28..0.29).contains(&size_ratio), "ratio {size_ratio}");
}

#[test]
fn should_split_the_contraction_into_summed_part_convolutions() {
    let shape = ProgramShape::new(256, 4, 2).expect("valid shape");
    let weights = quantized_weights(4, 256);

    let contents =
        encode_ml_program_projection(shape, MlProgramWeights::BlockwiseShiftScale(&weights), 4)
            .expect("program encodes");

    for expected_name in [
        "contraction_split".as_bytes(),
        b"activations_part_3",
        b"weight_part_3",
        b"partial_projection_3",
        b"contraction_sum_2",
    ] {
        assert!(contains_subsequence(
            &contents.model_document,
            expected_name
        ));
    }
    assert!(!contains_subsequence(
        &contents.model_document,
        b"contraction_sum_3"
    ));
}

#[test]
fn should_leave_out_the_split_for_a_single_convolution() {
    let shape = ProgramShape::new(64, 1, 1).expect("valid shape");
    let weights = quantized_weights(1, 64);

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::BlockwiseShiftScale(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    assert!(!contains_subsequence(
        &contents.model_document,
        b"contraction_split"
    ));
    assert!(!contains_subsequence(
        &contents.model_document,
        b"partial_projection"
    ));
}

#[test]
fn should_reject_a_part_that_would_cut_a_quantization_block() {
    let shape = ProgramShape::new(128, 1, 1).expect("valid shape");
    let weights = quantized_weights(1, 128);

    assert!(matches!(
        encode_ml_program_projection(shape, MlProgramWeights::BlockwiseShiftScale(&weights), 4),
        Err(ModelDocumentError::ContractionPartCutsQuantizationBlock {
            part_width: 32,
            block_input_columns: 64
        })
    ));
}

#[test]
fn should_store_each_parts_fp16_columns_in_part_order() {
    // One row of four weights, 1.0 2.0 3.0 4.0, split into two parts.
    let shape = ProgramShape::new(4, 1, 1).expect("valid shape");

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::Fp16(&[0x3c00, 0x4000, 0x4200, 0x4400]),
        2,
    )
    .expect("program encodes");

    // Part 0's data follows the first metadata record (offset 128), part 1's
    // the second (offset 256).
    assert_eq!(
        &contents.weight_blob_file[128..132],
        &[0x00, 0x3c, 0x00, 0x40]
    );
    assert_eq!(
        &contents.weight_blob_file[256..260],
        &[0x00, 0x42, 0x00, 0x44]
    );
    assert!(contains_subsequence(
        &contents.model_document,
        b"partial_projection_1"
    ));
}

#[test]
fn should_split_a_lookup_table_program_into_one_table_set_per_part() {
    let shape = ProgramShape::new(256, 2, 2).expect("valid shape");
    let weights = quantized_weights(2, 256);

    let contents = encode_ml_program_projection(shape, MlProgramWeights::LookupTable(&weights), 2)
        .expect("program encodes");

    // Per part: 256 four-bit indices (128 B) and 2 rows × 2 groups × 16
    // entries (128 B), each behind a metadata record.
    assert_eq!(
        contents.weight_blob_file.len(),
        64 + 2 * ((64 + 128) + (64 + 128))
    );
    assert!(contains_subsequence(
        &contents.model_document,
        b"weight_part_1"
    ));
}

#[test]
fn should_declare_eight_bit_shift_scale_codes_as_uint8() {
    let shape = ProgramShape::new(64, 1, 1).expect("valid shape");
    let weights = quantized_weights_in_layout(
        1,
        64,
        QuantizationLayout {
            code_bits: 8,
            block_output_rows: 1,
            block_input_columns: 64,
        },
    );

    let contents = encode_ml_program_projection(
        shape,
        MlProgramWeights::BlockwiseShiftScale(&weights),
        SINGLE_CONVOLUTION,
    )
    .expect("program encodes");

    // TensorType dataType UINT8 (31) and the 64 one-byte codes.
    assert!(contains_subsequence(
        &contents.model_document,
        &[0x08, 0x1f]
    ));
    assert_eq!(
        contents.weight_blob_file.len(),
        64 + (64 + 64) + (64 + 64) + (64 + 64)
    );
}

#[test]
fn should_reject_quantized_weights_of_another_shape() {
    let shape = ProgramShape::new(128, 2, 1).expect("valid shape");
    let weights = quantized_weights(1, 128);

    assert!(matches!(
        encode_ml_program_projection(
            shape,
            MlProgramWeights::BlockwiseShiftScale(&weights),
            SINGLE_CONVOLUTION
        ),
        Err(ModelDocumentError::QuantizedShapeMismatch {
            output_channels: 1,
            input_channels: 128
        })
    ));
}

#[test]
fn should_reject_fp16_weights_of_the_wrong_count_and_an_uneven_split() {
    let shape = ProgramShape::new(6, 1, 1).expect("valid shape");

    assert!(matches!(
        encode_ml_program_projection(shape, MlProgramWeights::Fp16(&[0; 5]), SINGLE_CONVOLUTION),
        Err(ModelDocumentError::WeightCountMismatch {
            expected: 6,
            actual: 5
        })
    ));
    assert!(matches!(
        encode_ml_program_projection(shape, MlProgramWeights::Fp16(&[0; 6]), 4),
        Err(ModelDocumentError::UnevenContractionSplit { .. })
    ));
    assert!(encode_ml_program_projection(shape, MlProgramWeights::Fp16(&[0; 6]), 0).is_err());
}
