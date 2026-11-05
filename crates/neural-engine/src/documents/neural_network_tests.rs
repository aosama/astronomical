use super::{ProjectionDocumentError, encode_projection_document, write_model_document};
use crate::projection_shape::ProjectionShape;

const SINGLE_CONVOLUTION: usize = 1;

fn contains_subsequence(document: &[u8], expected_bytes: &[u8]) -> bool {
    for window in document.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

#[test]
fn should_start_with_the_fp16_capable_specification_version() {
    let shape = ProjectionShape::new(2, 3, 4).expect("valid shape");

    let document = encode_projection_document(shape, &[0x3c00; 6], SINGLE_CONVOLUTION)
        .expect("document encodes");

    assert_eq!(&document[..2], &[0x08, 0x07]);
}

#[test]
fn should_name_the_features_the_native_bridge_binds() {
    let shape = ProjectionShape::new(2, 3, 4).expect("valid shape");

    let document = encode_projection_document(shape, &[0x3c00; 6], SINGLE_CONVOLUTION)
        .expect("document encodes");

    assert!(contains_subsequence(&document, b"activations"));
    assert!(contains_subsequence(&document, b"projected"));
}

#[test]
fn should_embed_every_weight_as_little_endian_fp16() {
    let shape = ProjectionShape::new(2, 1, 1).expect("valid shape");

    let document = encode_projection_document(shape, &[0x3c00, 0xc100], SINGLE_CONVOLUTION)
        .expect("document encodes");

    // WeightParams field 2 (float16Value), four bytes, 1.0 then -2.5.
    assert!(contains_subsequence(
        &document,
        &[0x12, 0x04, 0x00, 0x3c, 0x00, 0xc1]
    ));
}

#[test]
fn should_declare_channels_by_tokens_fp16_arrays() {
    let shape = ProjectionShape::new(5, 3, 7).expect("valid shape");

    let document =
        encode_projection_document(shape, &[0; 15], SINGLE_CONVOLUTION).expect("document encodes");

    // ArrayFeatureType: shape [1, 5, 1, 7] then dataType FLOAT16 (65552).
    let activations_array_type = [
        0x08, 0x01, 0x08, 0x05, 0x08, 0x01, 0x08, 0x07, 0x10, 0x90, 0x80, 0x04,
    ];
    assert!(contains_subsequence(&document, &activations_array_type));
}

#[test]
fn should_reject_a_weight_count_that_does_not_match_the_shape() {
    let shape = ProjectionShape::new(2, 3, 4).expect("valid shape");

    assert_eq!(
        encode_projection_document(shape, &[0; 5], SINGLE_CONVOLUTION),
        Err(ProjectionDocumentError::WeightCountMismatch {
            expected: 6,
            actual: 5
        })
    );
}

#[test]
fn should_split_the_contraction_into_summed_part_convolutions() {
    let shape = ProjectionShape::new(4, 1, 1).expect("valid shape");

    let document = encode_projection_document(shape, &[0x3c00, 0x4000, 0x4200, 0x4400], 2)
        .expect("document encodes");

    assert!(contains_subsequence(&document, b"contraction_split"));
    assert!(contains_subsequence(&document, b"activations_part_1"));
    assert!(contains_subsequence(&document, b"partial_projection_1"));
    assert!(contains_subsequence(&document, b"contraction_sum"));
    // The second part's weights are columns 2..4: 3.0 (0x4200), 4.0 (0x4400).
    assert!(contains_subsequence(
        &document,
        &[0x12, 0x04, 0x00, 0x42, 0x00, 0x44]
    ));
}

#[test]
fn should_keep_a_single_convolution_when_the_contraction_is_not_split() {
    let shape = ProjectionShape::new(4, 1, 1).expect("valid shape");

    let document =
        encode_projection_document(shape, &[0; 4], SINGLE_CONVOLUTION).expect("document encodes");

    assert!(!contains_subsequence(&document, b"contraction_split"));
}

#[test]
fn should_reject_a_contraction_split_that_does_not_divide_the_input_channels() {
    let shape = ProjectionShape::new(6, 1, 1).expect("valid shape");

    assert_eq!(
        encode_projection_document(shape, &[0; 6], 4),
        Err(ProjectionDocumentError::UnevenContractionSplit {
            input_channels: 6,
            contraction_part_count: 4
        })
    );
    assert!(encode_projection_document(shape, &[0; 6], 0).is_err());
}

#[test]
fn should_write_the_model_document_and_return_its_path() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");

    let model_path = write_model_document(model_directory.path(), &[1, 2, 3])
        .expect("the document should be written");

    assert_eq!(
        model_path,
        model_directory.path().join("projection.mlmodel")
    );
    assert_eq!(
        std::fs::read(&model_path).expect("the document should be readable"),
        [1, 2, 3]
    );
}

#[test]
fn should_write_an_empty_model_document() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");

    let model_path =
        write_model_document(model_directory.path(), &[]).expect("the document should be written");

    assert!(
        std::fs::read(&model_path)
            .expect("the document should be readable")
            .is_empty()
    );
}

#[test]
fn should_report_a_model_directory_that_does_not_exist() {
    let model_directory = tempfile::tempdir().expect("a scratch directory");

    let write_result = write_model_document(&model_directory.path().join("absent"), &[1]);

    assert!(write_result.is_err());
}
